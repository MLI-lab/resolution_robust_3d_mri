from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

def _ffeats(x: Tensor) -> Tensor:
    
    assert x.ndim == 2

    factor = np.sqrt(3)
    sinfeats = torch.sin(x) * factor
    cosfeats = torch.cos(x) * factor

    return torch.cat( (sinfeats, cosfeats), len(x.shape)-1 )

class FourierFeatureMap(nn.Module):

    def __init__(self,
        in_features: int,
        out_features: int,
        feats_scale: Optional[float] = None,
        init_sigma: Optional[float] = None,
        requires_bias: bool = False,
        requires_grad: bool = True
        ) -> nn.Module:

        super().__init__()
    
        self.feats_scale = feats_scale if not None else 1. 
        self.init_sigma = init_sigma

        self.linear = nn.Linear(in_features=in_features, 
            out_features=out_features, bias=requires_bias)
        self.linear.weight.requires_grad = requires_grad
        
        if self.init_sigma is not None: self._init()
        if requires_bias: self.linear.bias.requires_grad = requires_grad
    
    def _init(self, ) -> None:

        with torch.no_grad():
            self.linear.weight.normal_(self.init_sigma)
            if self.linear.bias is not None:
                self.linear.bias.normal_(self.init_sigma)

    def forward(self, x: Tensor) -> Tensor:

        x = self.linear(x)
        scaled_x = x * self.feats_scale
        return _ffeats(scaled_x)