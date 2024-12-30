from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor

from src.diffmodels.archs.std.unet import UNetModel

class PassThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, score: UNetModel) -> Tensor:
        return score(input1, input2)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        return grad_output, None, None
    
class ScoreWithIdentityGradWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.fn = PassThrough.apply

    def forward(self, input1, input2):
        return self.fn(input1, input2, self.module)
