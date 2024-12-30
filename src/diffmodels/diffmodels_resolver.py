from typing import Callable, Dict
from collections import OrderedDict

import torch
import logging

from .archs.std.unet import UNetModel
from .ema import ExponentialMovingAverage

# from src.utils.path_utils import get_path_by_cluster_name

from pathlib import Path
from omegaconf import OmegaConf


def create_dense_model(
    num_channels: int,
    in_channels: int,
    out_channels: int,
    num_res_blocks: int,
    channel_mult: str = '',
    use_checkpoint: bool = False,
    attention_resolutions: str = '16',
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0.,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    resamp_with_conv : bool = True,
    learn_sigma : bool = False,
    **kwargs
):

    logging.info(f"Unused kwargs: {kwargs}")
    
    attention_ds = []
    for res in attention_resolutions.split(","):
        #attention_ds.append(image_size // int(res))
        attention_ds.append(int(res)) # this is different now

    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else in_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=tuple(channel_mult),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        conv_resample=resamp_with_conv
    )

def create_model(name: str, params, arch_cfg = None) -> UNetModel:

    if name == "dense":
        return create_dense_model(**params)
    elif name == "depthwise_dm" or name == "sparse_fno":
        from .archs.infdm.model_sparse import SparseUNet
        if arch_cfg is not None:
            # there are certain parameters we overwrite
            if not "knn_neighbours" in params or arch_cfg.params.knn_neighbours != params["knn_neighbours"]:
                logging.warning(f"Overwriting knn_neighbours from {params['knn_neighbours'] if 'knn_neighbours' in params else 'unknown'} to {arch_cfg.params.knn_neighbours}")
                params["knn_neighbours"] = arch_cfg.params.knn_neighbours
            
            if not "kernel_interpolation_method" in params or arch_cfg.params.kernel_interpolation_method != params["kernel_interpolation_method"]:
                logging.warning(f"Overwriting kernel_interpolation_method from {params['kernel_interpolation_method'] if 'kernel_interpolation_method' in params else 'unknown'} to {arch_cfg.params.kernel_interpolation_method}")
                params["kernel_interpolation_method"] = arch_cfg.params.kernel_interpolation_method

        return SparseUNet(**params)
    else:
        raise ValueError(f"Unknown model name {name}")

def load_score_model(cfg: Dict, device : str, path_resolver : Callable) -> UNetModel:

    model_key, model_use_ema = cfg.model_key, cfg.model_use_ema
    # if there is a correct path available for loading the model, then load the pretrained model (otherwise load the randomly initalized one)
    if (model_use_ema and cfg.load_ema_params_from_path is not None) or (not model_use_ema and cfg.load_params_from_path is not None):
        
        score = None # will be loaded later
        assert model_key is not None, "model_key is not defined in the config file"

        if not model_use_ema:
            # first resolve the path to the model
            assert model_key in cfg.load_params_from_path, f"model_key {model_key} not found in load_ema_params_from_path"
            load_params_from_path = path_resolver(cfg.load_params_from_path[model_key])

            # the path points to the concrete model, before loading the model try accessing the arch
            hydra_train_config = Path(load_params_from_path).parent.joinpath('.hydra', 'config.yaml')
            kwargs_score = dict(OmegaConf.load(hydra_train_config).arch)
            # load the model (backward compatibility)
            if "name" not in kwargs_score:
                param_dict = {
                    "name" : "dense",
                    "params" : kwargs_score
                }
            else:
                param_dict = kwargs_score
            score = create_model(**param_dict, arch_cfg=cfg).to(device)

            try: 
                score.load_state_dict(
                    torch.load(load_params_from_path, map_location=device)
                )
            except: 
                state_dict = torch.load(load_params_from_path, map_location=device)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace('module.', '')] = v # remove 'module.' of DataParallel/DistributedDataParallel
                score.load_state_dict(new_state_dict)
            logging.info(f'model ckpt loaded from: {load_params_from_path}')

        else:
            assert model_key in cfg.load_ema_params_from_path, f"model_key {model_key} not found in load_params_from_path"
            load_ema_params_from_path = path_resolver(cfg.load_ema_params_from_path[model_key])

            # the path points to the concrete model, before loading the model try accessing the arch
            hydra_train_config = Path(load_ema_params_from_path).parent.joinpath('.hydra', 'config.yaml')
            try:
                kwargs_score = dict(OmegaConf.load(hydra_train_config).arch)
            except:
                kwargs_score = dict(OmegaConf.load(hydra_train_config).diffmodels.arch)
            # load the model (backward compatibility)
            if "name" not in kwargs_score:
                param_dict = {
                    "name" : "dense",
                    "params" : kwargs_score
                }
            else:
                param_dict = kwargs_score
            score = create_model(**param_dict, arch_cfg = cfg.arch if "arch" in cfg else None).to(device)

            ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
            ema.load_state_dict(torch.load(load_ema_params_from_path, map_location=device))
            ema.copy_to(score.parameters())
            logging.info(f'model ema ckpt loaded from: {load_ema_params_from_path}')

    else:
        logging.info("Creating model from scratch.")
        kwargs_score = dict(cfg.arch)
        score = create_model(**kwargs_score, arch_cfg=None).to(device)

    #score.dtype = torch.float32

    return score