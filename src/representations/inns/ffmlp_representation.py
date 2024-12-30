from typing import Optional, Any, Union, Tuple, Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
import copy
import logging

from src.representations.mesh import SliceableMesh
from src.representations.base_coord_based_representation import CoordBasedRepresentation

from src.representations.inns.layers.ffeatures import FourierFeatureMap
from src.representations.inns.layers.rfeatures import ReLULayer

def fetch_layer(name: str, 
            kwargs: Dict = {}) -> nn.Module:

    if name.lower() == 'fmap':

        out_features = kwargs['in_features']//2 if 'out_features' not in kwargs.keys() else kwargs['out_features']//2
        return FourierFeatureMap(
        in_features=kwargs['in_features'], 
        out_features=out_features, 
        feats_scale=kwargs['feats_scale'],
        init_sigma=kwargs['init_sigma'], 
        requires_grad=kwargs['requires_grad'],
        requires_bias=kwargs['requires_grad']
        )
    
    elif name.lower() == 'relu':
        out_features = kwargs['in_features'] if 'out_features' not in kwargs.keys() else kwargs['out_features']
        return ReLULayer(
        in_features=kwargs['in_features'], 
        out_features=out_features, 
        feats_scale=kwargs['feats_scale'],
        init_sigma=kwargs['init_sigma'], 
        normalizerelu=kwargs['normalizerelu'],
        requires_bias=kwargs['requires_grad']
        )
    else: 
        raise NotImplementedError


class FFmlpRepresentation(CoordBasedRepresentation, nn.Module):
    
    def __init__(self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        width: int,
        requires_bias: bool = False,
        first_layer_feats_scale: float = 30.0,
        init_sigma: float = 1.0,
        final_sigma: float = 0.01,
        act_type = 'fmap',
        first_layer_trainable: bool = False,
        first_layer_fmap: bool = True,
        first_layer_init_sigma : Optional[float] = None,
        normalizerelu: bool = True,
        eps: float = 1e-4,
        device: Optional[Any] = None,
        warm_start: Optional[Tensor] = None,
        warm_start_mesh : Optional[Tensor] = None,
        warm_start_cfg : Optional[DictConfig] = None
        ):

        super().__init__()
        
        self.arch = []
        #self.mesh = mesh
        self.eps = eps
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        # init layers
        self.arch.append(fetch_layer(name='fmap' if first_layer_fmap else 'relu', kwargs={
            'in_features': in_features, 
            'out_features': width, 
            'feats_scale': first_layer_feats_scale,
            'init_sigma': first_layer_init_sigma if first_layer_init_sigma is not None else 1/in_features,
            'requires_grad': first_layer_trainable, 
            'requires_bias': requires_bias
            })
        )

        # hidden layers
        hidden_kwargs = {
            'in_features': width,
            'feats_scale': 1.0, 
            'init_sigma': init_sigma, 
            'requires_grad': True,
            'normalizerelu': normalizerelu
        }
        for _ in range(num_hidden_layers):
            self.arch.append(fetch_layer(name=act_type, kwargs=hidden_kwargs))
        
        # last layer
        final_linear = nn.Linear(width, out_features, bias=True)
        final_linear.weight.data.normal_(std=final_sigma)
        final_linear.bias.data.normal_(std=self.eps)
    
        self.arch.append(final_linear)
        self.net = nn.Sequential(*self.arch).to(self.device)

        # new: warm_start
        if warm_start is not None:
            logging.info("Performing warmstart init.")
            from tqdm import tqdm
            optimizer = torch.optim.Adam(self.parameters(), lr=warm_start_cfg.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
            from src.reconstruction.utils.metrics import PSNR
            loss_fn = nn.MSELoss()
            with tqdm(range(warm_start_cfg.iterations), desc='warmstart') as pbar:
                for i in pbar:
                    optimizer.zero_grad()
                    x = self.forward(warm_start_mesh)
                    loss = loss_fn(x, warm_start)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)

                    psnr = PSNR(x.detach(), warm_start.detach())
                    pbar.set_description(f'psnr={psnr:.1f}', refresh=False)

                    if warm_start_cfg.psnr_threshold is not None:
                        if psnr > warm_start_cfg.psnr_threshold:
                            logging.info(f"Warmstart phase finished early due to PSNR threshold ({psnr} > {warm_start_cfg.psnr_threshold}).")
                            break

    def get_optimizer_params(self) -> Tuple:
        return self.parameters()

    def forward(self, mesh: SliceableMesh) -> Tensor:
        mesh_tensor = mesh.get_coord_mesh().to(self.device)
        return self.net(
                mesh_tensor.reshape(-1, self.in_features) # feed in (B, 3) -> output is (B, C)
            ).reshape(*mesh_tensor.shape[:-1], self.out_features) # reshape to (Z, Y, X, C)
    
    def forward_splitted(self, mesh : SliceableMesh, custom_device : Optional[Any] = None, split : int = 1) -> Tensor:
        """
            Same as forward, but with the option to split the input tensor into smaller parts to tradeoff GPU memory vs time
        """
        device = custom_device if custom_device is not None else self.device
        net = self.net if device is self.device else copy.deepcopy(self.net).cpu().to(device) # need this cpu inbetween here
        mesh_tensor = mesh.get_coord_mesh().to(device)
        mesh_tensor_vsplits = mesh_tensor.reshape(-1, self.in_features).vsplit(split)
        net_out_vsplits = [net(mesh_tensor_part) for mesh_tensor_part in mesh_tensor_vsplits]
        return torch.vstack(net_out_vsplits).reshape(*mesh_tensor.shape[:-1], self.out_features)