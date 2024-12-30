from torch import Tensor

from .base_target_trafo import BaseTargetTrafo
from .base_prior_trafo import BasePriorTrafo

class IdentityTrafo(BasePriorTrafo, BaseTargetTrafo):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def trafo_inv(self, x: Tensor) -> Tensor:
        return x
