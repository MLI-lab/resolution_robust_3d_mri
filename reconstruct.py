from itertools import islice
from functools import partial

from typing import Optional

import os
import math

import torch.nn as nn

import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np

from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.problem_trafos.trafo_resolver import (
    get_fwd_trafo, get_dataset_trafo,
    get_target_trafo, get_prior_trafo)

from src.diffmodels.sampler.sampler_resolver import get_sampler
from src.diffmodels.sde import SDE
from src.reconstruction.posterior_sampling.conditioner_resolver import get_conditioning_method
from src.representations.representation_resolver import (
    get_representation,
    get_mesh,
    get_mesh_from_model,
    get_slice_method
)
from src.datasets import dataset_resolver

from src.reconstruction.variational.fit import fit
from src.reconstruction.utils.pass_through import ScoreWithIdentityGradWrapper
from src.reconstruction.variational.var_objectives import get_variational_objective
from src.utils.wandb_utils import wandb_kwargs_via_cfg
from src.utils.device_utils import get_free_cuda_devices
from src.utils.path_utils import get_path_by_cluster_name
from src.problem_trafos.utils.bart_utils import import_bart

from src.diffmodels.diffmodels_resolver import load_score_model
from src.diffmodels import load_sde_model
from src.sample_logger.sample_logger_resolver import get_sample_logger
from src.representations.fixed_grid_representation import FixedGridRepresentation

import hydra

@hydra.main(config_path='hydra', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:

    OmegaConf.resolve(cfg)
    wandb_kwargs = wandb_kwargs_via_cfg(cfg)
    devices : list[str] = get_free_cuda_devices(**cfg.cuda_devices)
    device = devices[0]

    # importing bart, the path has to match the path from the config
    import_bart(os.path.join(hydra.utils.get_original_cwd(), 'libs', 'bart-0.6.00'))

    with wandb.init(**wandb_kwargs) as run:

        dtype = torch.get_default_dtype()

        # trafo: object to measurement data (e.g. complex fourier data or sinogram)
        fwd_trafo = get_fwd_trafo(**cfg.problem_trafos.fwd_trafo)
        # trafo: object to target data (e.g. magnitude images)
        target_trafo = get_target_trafo(**cfg.problem_trafos.target_trafo)
        # trafo: object to prior (e.g. magnitude images)
        prior_trafo = get_prior_trafo(**cfg.problem_trafos.prior_trafo)
        # trafo: dataset trafo (preprocessing loaded volumes)
        dataset_trafo = get_dataset_trafo(**cfg.problem_trafos.dataset_trafo,
            provide_pseudoinverse=True, provide_measurement=True, device=device)

        # resolving paths, e.g. for the datasets and loaded models, depending on the cluster
        path_resolver = partial(get_path_by_cluster_name, cfg=cfg)

        # loading the datasets
        dataset = dataset_resolver.get_dataset(
            **cfg.dataset, dataset_trafo=dataset_trafo, path_resolver=path_resolver)

        # loading score model and sde
        score : Optional[nn.Module] = None
        sde : Optional[SDE] = None
        if cfg.reconstruction.use_score_regularisation: 
            score = load_score_model(cfg.diffmodels, device=device, path_resolver=path_resolver)
            sde = load_sde_model(cfg.diffmodels)
            if wandb.run is not None:
                wandb.run.summary['num_params_score'] = sum(p.numel() for p in score.parameters() if p.requires_grad)
            if cfg.reconstruction.use_score_pass_through:
                score = ScoreWithIdentityGradWrapper(module=score)

        if len(dataset) < cfg.reconstruction.num_images:
            logging.warning(f'Only {len(dataset)} images available in dataset.')
            num_samples = len(dataset)
        else:
            num_samples = cfg.reconstruction.num_images

        # create and init logging
        sample_logger = get_sample_logger(device=device, devices=devices, **cfg.sample_logger,
            fwd_trafo=fwd_trafo, target_trafo=target_trafo)
        sample_logger.init_run(num_samples=num_samples)

        for i, data_sample in enumerate(tqdm(islice(DataLoader(dataset), cfg.reconstruction.dataset_start, cfg.reconstruction.dataset_start+num_samples),total=num_samples)):

            observation, ground_truth, filtbackproj, attrs = data_sample

            observation = observation.to(dtype=dtype, device=device).squeeze()
            filtbackproj = filtbackproj.to(dtype=dtype, device=device).squeeze()
            ground_truth = ground_truth.to(dtype=dtype, device=device).squeeze()

            rep_shape = filtbackproj.shape
            prior_shape = prior_trafo(filtbackproj).shape

            base_mesh_shape = rep_shape[:-1] if cfg.representation.arch.out_features > 1 else rep_shape 
            if cfg.representation.mesh_data.matrix_size is None:
                logging.info(f"Matrix-size is None, take base_mesh_shape: {base_mesh_shape}.")
                cfg.representation.mesh_data.matrix_size = tuple(base_mesh_shape)
            elif base_mesh_shape != cfg.representation.mesh_data.matrix_size:
                logging.error(f"base_mesh_shape: {base_mesh_shape} and matrix_size: {cfg.representation.mesh_data.matrix_size} do not match.  Use base_mesh_shape for mesh creation.")
                cfg.representation.mesh_data.matrix_size = tuple(base_mesh_shape)
            # mesh_data_con = get_mesh(cfg.representation.mesh_data_name, cfg.representation.mesh_data, device=device)
            mesh_data_con = get_mesh(cfg.representation.mesh_data, device=device)

            logging.info("Calibrating trafo")
            fwd_trafo.calibrate(observation, attrs)

            scaling_factor : float = 1.0
            if cfg.reconstruction.rescale_observation:
                scaling_factor = math.sqrt(float(np.prod(rep_shape).item())) / observation.detach().cpu().norm() * float(cfg.reconstruction.constant_scaling_factor)
            else:
                scaling_factor = cfg.reconstruction.constant_scaling_factor

            sample_logger.init_sample_log(
                observation=observation,
                ground_truth=ground_truth,
                filtbackproj=filtbackproj,
                sample_nr=i,
                scaling_factor = scaling_factor,
                mesh=mesh_data_con
            )

            final_representation : CoordBasedRepresentation

            if cfg.reconstruction.method == 'sampling':
                ####################################
                ### Sampling based methods
                ####################################

                if cfg.reconstruction.sampling is not None:
                    cond_method = get_conditioning_method(
                        fwd_trafo=fwd_trafo,
                        prior_trafo=prior_trafo,
                        im_shape=rep_shape,
                        observation=observation * scaling_factor,
                        sde=sde,
                        **cfg.reconstruction.sampling,
                    )
                else:
                    cond_method = None

                sampler = get_sampler(
                    sample_logger=sample_logger,
                    conditioning_method=cond_method,
                    fwd_trafo=fwd_trafo,
                    prior_trafo=prior_trafo,
                    score=score,
                    sde=sde,
                    device=device,
                    im_shape=prior_shape,
                    **cfg.diffmodels.sampler
                )
                sample = sampler.sample().detach()
                final_representation = FixedGridRepresentation(
                    in_shape=tuple(sample.shape[:-1]),
                    out_features=sample.shape[-1],
                    warm_start=sample,
                )

            elif cfg.reconstruction.method == 'variational':
                ####################################
                ### Variational methods
                ####################################

                # datacon mesh is already initialized
                mesh_prior_reg = None
                if cfg.representation.mesh_prior_use_same_as_data:
                    logging.warning("Using same mesh for data and prior.")
                    mesh_prior_reg = get_mesh(
                        mesh_cfg=cfg.representation.mesh_data,
                        device=device,
                    )
                elif (
                    cfg.representation.mesh_prior.matrix_size is not None
                    and cfg.representation.mesh_prior.field_of_view is not None
                ):
                    logging.info(
                        f"Taking fixed mesh of size: {cfg.representation.mesh_prior.matrix_size} and fov: {cfg.representation.mesh_prior.field_of_view} for prior."
                    )
                    mesh_prior_reg = get_mesh(
                        cfg.representation.mesh_prior,
                        device=device,
                    )
                else:
                    logging.info("Trying to derive mesh from prior.")
                    mesh_prior_reg = get_mesh_from_model(
                        mesh_cfg = cfg.representation.mesh_prior,
                        device=device,
                        model_key=cfg.diffmodels.model_key,
                        mesh_data_per_model=cfg.diffmodels.mesh_data_per_model,
                    )

                initialise_with = None
                if cfg.reconstruction.variational.fitting.use_filterbackproj_as_init: 
                    initialise_with = torch.clone(filtbackproj) * scaling_factor
                    logging.info("initialise with pseudoinverse")
                elif cfg.reconstruction.variational.fitting.use_l1wavelet_as_init:
                    from src.problem_trafos.utils.bart_utils import compute_l1_wavelet_solution
                    l1_wavelet_solution = compute_l1_wavelet_solution(observation, attrs["sens_maps"], reg_param=4e-4)
                    initialise_with = l1_wavelet_solution * scaling_factor
                    logging.info("initialise with l1 wavelet")
                filtbackproj = None; torch.cuda.empty_cache()

                representation = get_representation(
                    representation_cfg=cfg.representation,
                    mesh_data = mesh_data_con,
                    mesh_prior = mesh_prior_reg,
                    initialise_with = initialise_with,
                    device=device,
                )

                if wandb.run is not None:
                    params = representation.get_optimizer_params()
                    if isinstance(params, list):
                        wandb.run.summary['num_params_representation'] = sum(
                            sum(p.numel() for p in group['params'] if p.requires_grad) 
                            for group in params
                        )
                    else:
                        wandb.run.summary['num_params_representation'] = sum(
                            p.numel() for p in params if p.requires_grad
                        )

                slice_method_data_con = get_slice_method(**cfg.reconstruction.slice_methods.data_con)
                slice_method_prior_reg = get_slice_method(**cfg.reconstruction.slice_methods.prior_reg)

                var_objective = get_variational_objective(
                    # base objective args
                    observation=observation * scaling_factor,
                    mesh_data_con=mesh_data_con,
                    mesh_data_reg=mesh_prior_reg,
                    fwd_trafo=fwd_trafo,
                    prior_trafo=prior_trafo,
                    steps_data_con=cfg.reconstruction.variational.fitting.optimizer.gradient_acc_steps_data_con,
                    steps_data_reg=cfg.reconstruction.variational.fitting.optimizer.gradient_acc_steps_prior_reg,
                    slice_method_data_con=slice_method_data_con,
                    slice_method_prior_reg=slice_method_prior_reg,
                    outer_iterations_max = cfg.reconstruction.variational.fitting.optimizer.iterations,
                    score=score, 
                    sde=sde,
                    cfg_regularization = cfg.reconstruction.variational.regularization
                )

                final_representation = fit(
                    representation=representation, 
                    var_objective=var_objective,
                    cfg_fitting = cfg.reconstruction.variational.fitting,
                    sample_logger=sample_logger,
                )

            else:
                raise NotImplementedError(f'Reconstruction method {cfg.reconstruction.method} not implemented.')

            sample_logger.close_sample_log(representation=final_representation)

        sample_logger.close_run()

if __name__ == '__main__':
    coordinator()  
# %%
