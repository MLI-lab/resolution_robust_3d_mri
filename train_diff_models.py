"""
    Module: train_diff_models.py
"""
import logging
import os
from functools import partial

from typing import Callable

import wandb

import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
import hydra

from src.diffmodels import load_score_model
from src.diffmodels import score_model_trainer, load_sde_model
from src.utils.wandb_utils import wandb_kwargs_via_cfg
from src.utils.device_utils import get_free_cuda_devices
from src.utils.path_utils import get_path_by_cluster_name

from src.datasets.dataset_resolver import get_dataset
from src.problem_trafos.trafo_resolver import (get_dataset_trafo,
    get_prior_trafo)

from src.diffmodels.trainer.in_memory_dataset import cache_iterable_in_memory

from src.diffmodels.sampler.sampler_resolver import get_sampler
from src.sample_logger.sample_logger_resolver import get_sample_logger

def get_dataloader(cfg_dataset, dataset_trafo,  cfg_dl, fold, device : str, path_resolver : Callable):
    
    # first check if we need to cache the dataset in gpu
    cache_in_gpu = False
    cache_device = None
    if cfg_dl.cache_dataset:
        cache_in_gpu = cfg_dl.cache_dataset_in_gpu
        cache_device = device if cache_in_gpu  else "cpu"

    # first try to load the dataset from the fs cache
    dataset = None
    if cfg_dl.cache_dataset and cfg_dl.cache_dataset_load_from_disk:
        if os.path.exists(cfg_dl.cache_dataset_disk_path):
            dataset = torch.load(cfg_dl.cache_dataset_disk_path, map_location=cache_device)
        else:
            logging.warning(f"Could not find cached dataset at {cfg_dl.cache_dataset_disk_path} -> Loading it from scratch.")

    # create the dataset as usual when it has not been loaded from disk
    if dataset is None:
        iterable_dataset = get_dataset(**cfg_dataset, fold_overwrite=fold, dataset_trafo=dataset_trafo, path_resolver=path_resolver)

        if cfg_dl.cache_dataset:
            dataset = cache_iterable_in_memory(
                iterable_ds=iterable_dataset, use_tqdm=True, device=cache_device, repeat_dataset=cfg_dl.cache_dataset_repeats
                )

            if cfg_dl.cache_dataset_store_on_disk:
                torch.save(dataset, cfg_dl.cache_dataset_disk_path)
        else:
            dataset = iterable_dataset

    # create the dataloader
    if cfg_dl.use_batch_sampler_same_shape:
        from src.diffmodels.trainer.batch_sampler_same_shape import BatchSamplerSameShape
        sampler = BatchSamplerSameShape(dataset,
            shuffle        =    cfg_dl.shuffle,
            batch_size     =    cfg_dl.batch_size,
            group_shape_by =    cfg_dl.group_shape_by)

        dataloader = DataLoader(
            dataset,
            pin_memory=False,
            num_workers=cfg_dl.num_workers if not cache_in_gpu else 0,
            batch_sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg_dl.batch_size,
            shuffle=cfg_dl.shuffle,
            num_workers=cfg_dl.num_workers if not cache_in_gpu else 0,
            pin_memory=False
        )

    return dataloader

@hydra.main(config_path='hydra', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:

    OmegaConf.resolve(cfg)
    wandb_kwargs = wandb_kwargs_via_cfg(cfg)
    device = get_free_cuda_devices(**cfg.cuda_devices)[0]

    logging.getLogger().setLevel(logging.INFO)

    with wandb.init(**wandb_kwargs) as run:

        # resolving paths, e.g. for the datasets and loaded models, depending on the cluster
        path_resolver = partial(get_path_by_cluster_name, cfg=cfg)

        # get prior trafo
        prior_trafo = get_prior_trafo(**cfg.problem_trafos.prior_trafo)

        # dataset trafo
        dataset_trafo = get_dataset_trafo(**cfg.problem_trafos.dataset_trafo,
            provide_pseudoinverse=False, provide_measurement=False)

        # dataloader via the diffmodels dataloader configs & the dataset
        dataloader_train = get_dataloader(cfg_dataset=cfg.dataset,
            dataset_trafo = dataset_trafo, cfg_dl=cfg.diffmodels.train,
            fold="train", device=device, path_resolver=path_resolver)
        dataloader_val = get_dataloader(cfg_dataset=cfg.dataset,
            dataset_trafo = dataset_trafo, cfg_dl=cfg.diffmodels.val,
            fold="val", device=device, path_resolver=path_resolver)

        # load the sde model and the score model
        sde = load_sde_model(cfg.diffmodels)
        score = load_score_model(cfg.diffmodels, device=device, path_resolver=path_resolver)
        if wandb.run is not None:
            wandb.run.summary['num_params_score'] = sum(p.numel() for p in score.parameters() if p.requires_grad)

        # setup logging 
        sample_logger = get_sample_logger(**cfg.sample_logger)
        sampler = get_sampler(
                score=score,
                sde=sde,
                im_shape = prior_trafo(next(iter(dataloader_val))).shape,
                device=device,
                sample_logger=sample_logger,
                prior_trafo=prior_trafo,
                **cfg.diffmodels.sampler
        )

        # start training
        score_model_trainer(    
            score=score,
            sde=sde,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            sample_logger=sample_logger,
            sampler=sampler,
            optim_kwargs=cfg.diffmodels.train,
            val_kwargs=cfg.diffmodels.val,
            prior_trafo=prior_trafo,
            device=device
        )

        torch.save(score.state_dict(), 'last_model.pt')

if __name__ == '__main__':
    coordinator()  