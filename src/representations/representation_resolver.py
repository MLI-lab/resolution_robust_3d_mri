from typing import Optional, Any, Union, Tuple
from abc import ABC, abstractclassmethod

import torch.nn as nn
import numpy as np 

from omegaconf import DictConfig

from torch import Tensor
import logging

from typing import Dict, Optional

from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh
from src.representations.slice_methods import RandomAverageSlabsWithSliceableMesh
from src.representations.fixed_grid_representation import FixedGridRepresentation
from src.representations.grid_sampled_representation import GridResamplingRepresentation
from src.representations.gaussian.gaussian_representation import GaussianRepresentation
from src.representations.inns.ffmlp_representation import FFmlpRepresentation

def get_representation(
    representation_cfg: DictConfig,
    mesh_data : SliceableMesh,
    mesh_prior : SliceableMesh,
    initialise_with: Optional[Tensor] = None,
    device: Optional[str] = None,
    ) -> CoordBasedRepresentation:

    if mesh_prior is not None and representation_cfg.use_high_res_mesh_for_rep:
        assert mesh_data.field_of_view == mesh_prior.field_of_view, "Field of view of data and prior mesh must be the same in current implementation."
        mesh_rep= mesh_data if np.prod(mesh_data.matrix_size) > np.prod(mesh_prior.matrix_size) else mesh_prior
    elif mesh_prior is None:
        logging.warning("No prior mesh provided. Using data mesh.")
        mesh_rep = mesh_data
    else:
        logging.info("Using data mesh for rep.")
        mesh_rep = mesh_data

    if representation_cfg.name == 'parametric':
        kwargs = {
            'in_features': representation_cfg.arch.in_features,
            'out_features': representation_cfg.arch.out_features,
            'num_hidden_layers': representation_cfg.arch.num_hidden_layers,
            'normalizerelu': representation_cfg.arch.normalizerelu,
            'first_layer_feats_scale': representation_cfg.arch.first_layer_feats_scale,
            'final_sigma': representation_cfg.arch.final_sigma, 
            'act_type': representation_cfg.arch.act_type, 
            'first_layer_trainable': representation_cfg.arch.first_layer_trainable,
            'first_layer_fmap': representation_cfg.arch.first_layer_fmap, 
            'first_layer_init_sigma': representation_cfg.arch.first_layer_init_sigma,
            'init_sigma': representation_cfg.arch.init_sigma,
            'eps': representation_cfg.arch.eps
            }
        
        if representation_cfg.arch.width_from_mesh:
            #width = int(mesh_rep.matrix_size[0] / np.sqrt(2) / 2 ) * 2 # what is the reasoning here?
            width = int( np.sqrt(np.prod(mesh_rep.matrix_size)) / np.sqrt(3) / 2 ) * 2
        else:
            width = representation_cfg.arch.width

        image_parametrisation = FFmlpRepresentation(
            width=width,
            device=device,
            **kwargs,
            warm_start=initialise_with,
            warm_start_mesh=mesh_data,
            warm_start_cfg = representation_cfg.warmstart_optim
        )

    elif representation_cfg.name == 'identity':
        assert mesh_rep.matrix_size == mesh_data.matrix_size, "Meshes must be the same for identity representation (set use_same_mesh option)."
        image_parametrisation = FixedGridRepresentation(in_shape=mesh_rep.matrix_size,out_features=representation_cfg.arch.out_features, warm_start=initialise_with, device=device)
    elif representation_cfg.name == 'grid_sampled':
        image_parametrisation = GridResamplingRepresentation(rep_mesh=mesh_rep, out_features=representation_cfg.arch.out_features, warm_start=initialise_with, warm_start_mesh=mesh_data, device=device, interpolation_mode=representation_cfg.interpolation_mode, padding_mode=representation_cfg.padding_mode, align_corners=representation_cfg.align_corners)
    elif representation_cfg.name == 'gaussian':
        image_parametrisation = GaussianRepresentation(warm_start=initialise_with, warm_start_mesh=mesh_data, warm_start_cfg=representation_cfg.warmstart_optim,  device=device, model_params=representation_cfg.model_params, opt_params=representation_cfg.opt_params)
    else: 
        raise NotImplementedError('Unsupported name')
    
    return image_parametrisation

def get_mesh(
        mesh_cfg,
        device: Optional[Any] = None
    ) -> SliceableMesh:
    return SliceableMesh(**mesh_cfg, device=device)
    # name : str,
    # if name == "sliceable_mesh":
        # return SliceableMesh(**mesh_cfg, device=device)
    # elif name == "dynamic_sliceable_mesh":
        # return DynamicSliceableMesh(**mesh_cfg, device=device)

def get_mesh_from_model(
        mesh_cfg,
        model_key : str,
        mesh_data_per_model,
        device: Optional[Any] = None
    ) -> SliceableMesh:
    # name : str,

    if model_key not in mesh_data_per_model:
        raise ValueError(f"Model key {model_key} not found in mesh_data_per_model.")
    else:
        mesh_compl = mesh_data_per_model[model_key]
        mesh_cfg.matrix_size = mesh_compl.matrix_size
        mesh_cfg.field_of_view = mesh_compl.field_of_view
        logging.info(f"Using mesh from model: {model_key} with matrix_size: {mesh_cfg.matrix_size} and field_of_view: {mesh_cfg.field_of_view}")

    return get_mesh(mesh_cfg, device=device)  

def get_slice_method(name : str, **slice_cfg):
    if name == "rnd_slicing":
        return RandomAverageSlabsWithSliceableMesh(**slice_cfg)
    elif name == 'None':
        return None
    else:
        raise ValueError(f"Unknown slice method {name}")