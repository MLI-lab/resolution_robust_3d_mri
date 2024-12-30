from __future__ import annotations
from typing import Optional, Any
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any
from torch import Tensor

from .base_fwd_trafo import BaseFwdTrafo

class IdentityTrafo(BaseFwdTrafo):

    def __init__(self):
        super().__init__()

    def calibrate(self, observation: Tensor, calib_params) -> None:
        pass

    def trafo(self, x: Tensor, slice_inds : Optional[Tensor] = None, slice_axis : Optional[int]= None) -> Tensor:
        return x

    def trafo_adjoint(self, y: Tensor) -> Tensor:
        return y