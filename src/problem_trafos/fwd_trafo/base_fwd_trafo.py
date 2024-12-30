"""
Provides :class:`BaseRayTrafo`.
"""

from typing import Dict
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor
import numpy as np


class BaseFwdTrafo(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    def calibrate(self, y: Tensor, calib_params : Dict):
        """
        Used to calibrate the forward map, e.g. to calculate sensitivity maps for MRI.
        """
        pass

    @abstractmethod
    def trafo(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection.
        """
        raise NotImplementedError

    @abstractmethod
    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        """
        """
        raise NotImplementedError

    def fbp(self, observation: Tensor) -> Tensor:
        """
        Apply a filtered back-projection.

        Parameters
        ----------
        observation : :class:`torch.Tensor`
            Projection values.
            Shape for 2D geometries: ``(batch, channels, angles, det_cols)``.
            Shape for 3D geometries: ``(batch, channels, det_rows, angles, det_cols)``.

        Returns
        -------
        x : :class:`torch.Tensor`
            Filtered back-projection.
            Shape for 2D geometries: ``(batch, channels, im_0, im_1)``.
            Shape for 3D geometries: ``(batch, channels, im_0, im_1, im_2)``.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """See :meth:`trafo`."""
        return self.trafo(x)
