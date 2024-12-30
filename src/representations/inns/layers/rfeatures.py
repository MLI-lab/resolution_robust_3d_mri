from typing import Optional
import torch 
import torch.nn as nn
import numpy as np

from torch import Tensor

class ReLULayer(nn.Module):    
    def __init__(self, 
        in_features: int, 
        out_features: int,
        feats_scale: Optional[float] = None, 
        init_sigma: float = 1.,
        normalizerelu: bool = False,
        requires_bias: bool = True,
        eps: float = 1e-5
        ) -> None:
        super().__init__()

        self.feats_scale = feats_scale if not None else 1. 
        self.init_sigma = init_sigma
        self.out_features = out_features
        self.linear = nn.Linear(in_features=in_features, 
                    out_features=out_features, bias=requires_bias)
        
        self._init_layer()
        self.normalizerelu = normalizerelu
        self.eps = eps
    
    def _init_layer(self, ) -> None:
        with torch.no_grad():
            self.linear.weight.normal_(std=self.init_sigma)
            if self.linear.bias is not None:
                self.linear.bias.normal_(std=1e-6)

    def forward(self, x: Tensor):

        x = self.feats_scale * self.linear(x)
        if self.normalizerelu == False:
            return np.sqrt(2 / self.out_features) * torch.relu(x)
        else:
            #feats = np.sqrt(2/self.out_features) * torch.nn.functional.relu6(x)
            feats = torch.nn.functional.relu6(x)
            scale = ( 1 / ( torch.std(feats, unbiased=False, dim=-1) + self.eps) )[:, None]
            mean = torch.mean(feats, dim=-1)[:, None]            
            return (feats - mean) * scale
        