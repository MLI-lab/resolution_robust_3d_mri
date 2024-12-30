"""
Provides :class:`MatmulRayTrafo`.
"""

from __future__ import annotations  # postponed evaluation, to make ArrayLike look good in docs
from typing import Union, Optional, Callable, Tuple, Any
try:
    from numpy.typing import ArrayLike
except ImportError:
    ArrayLike = Any
import torch
from torch import Tensor
import numpy as np
import scipy.sparse
import logging

from .base_fwd_trafo import BaseFwdTrafo
import fastmri
import fastmri.data.subsample
import fastmri.data.transforms
import torch.nn.functional as F
import torchvision.transforms as T

from src.utils.fftn3d import fft3c, ifft3c

from sigpy.mri.samp import poisson, radial, spiral

class SubsampledFourierTrafo3D(BaseFwdTrafo):
    """
    Subsampled Fourier ransform implemented by (sparse) matrix multiplication.

    """

    def __init__(self,
            mask_enabled : bool,
            mask_type : str,
            mask_accelerations : Tuple[int],
            mask_center_fractions : Tuple[float],
            mask_seed : int,
            include_sensitivitymaps : bool,
            sensitivitymaps_complex : bool,
            sensitivitymaps_fillouter : bool,
            wrapped_2d_mode : bool
        ):
        super().__init__()

        self.mask_seed = mask_seed
        self.mask_enabled = mask_enabled
        self.mask_center_fractions = mask_center_fractions
        self.mask_accelerations = mask_accelerations
        self.mask_type = mask_type
        self.wrapped_2d_mode = wrapped_2d_mode

        self.include_sensitivitymaps = include_sensitivitymaps
        self.sensitivitymaps_complex = sensitivitymaps_complex
        self.sensitivitymaps_fillouter = sensitivitymaps_fillouter


    def _set_mask(self, obs_shape, calib_params) -> None:
        if self.mask_enabled:
            if self.mask_type == 'dataset':
                assert "mask" in calib_params, "mask not found in calibration parameters"
                self.mask = calib_params["mask"]
            elif self.mask_type == 'Poisson2D':
                self.mask = torch.from_numpy(poisson(obs_shape[-3:-1], self.mask_accelerations, seed=self.mask_seed).astype(np.float32)).unsqueeze(dim=-1)
            else:
                mask_class = fastmri.data.subsample.RandomMaskFunc if self.mask_type == 'random' else fastmri.data.subsample.EquispacedMaskFractionFunc
                mask_func = mask_class(center_fractions=self.mask_center_fractions, accelerations=[self.mask_accelerations], seed=self.mask_seed)
                shape = (1,) * len(obs_shape[:-3]) + tuple(obs_shape[-3:])
                self.mask, num_low_frequencies = mask_func(shape, offset=None, seed=self.mask_seed)
        #else:
            #self.mask = None

    def calibrate(self, observation: Tensor, calib_params) -> None:

        self._set_mask(observation.shape, calib_params)
        if self.mask_enabled:
            self.mask = self.mask.to(observation.device)

        if self.include_sensitivitymaps:
            
            if calib_params is not None:
                logging.info("Using calibration parameters provided")
                if calib_params["sens_maps"].shape[0] != 1:
                    S = torch.view_as_real(
                        torch.from_numpy(calib_params["sens_maps"]).moveaxis(-1,0)
                    )
                else:
                    S = torch.view_as_real(
                            calib_params["sens_maps"].squeeze(0).moveaxis(-1,0)
                        )
            else:
                from src.problem_trafos.utils.bart_utils import compute_sens_maps_3d
                path = ""; enable_cache = False
                import os
                if enable_cache and os.path.exists(path):
                    logging.warning(f"Loading sensmaps from file: {path}")
                    with open(path, 'rb') as f:
                        sens_maps = np.load(f)
                else:
                    logging.warning("Multiple slices, calculations may take some time")
                    assert not self.wrapped_2d_mode, "wrapped_2d_mode not supported for 3D sensemap calculation"
                    sens_maps = compute_sens_maps_3d(observation[0])

                    if enable_cache:
                        logging.warning("Saving maps")
                        with open(path, 'wb') as f:
                            np.save(f, sens_maps)

                sens_maps = np.moveaxis(sens_maps, -1, 0)
                S = sens_maps.copy() 
                S = np.stack((S.real, S.imag), axis=-1)
                S = torch.from_numpy(S)

            if self.sensitivitymaps_complex:
                S = torch.view_as_complex(S)
            else:
                S = torch.view_as_real(S)

            self.sense_matrix = S.to(observation.device)
            self.sense_matrix_normalization_constant = S.abs().square().sum(dim=-4).sqrt().to(observation.get_device())

            # self.sense_matrix_normalization_constant[self.sense_matrix_normalization_constant == 0.0] = 1.0

            if self.sensitivitymaps_fillouter:
                cond = (self.sense_matrix_normalization_constant == 0.0).expand(self.sense_matrix.shape)
                self.sense_matrix[cond] = 1.0 / np.sqrt(self.sense_matrix.shape[0])

            S = None

            torch.cuda.empty_cache()

        else:
            self.sense_matrix = None

    def trafo(self, x: Tensor, slice_inds : Optional[Tensor] = None, slice_axis : Optional[int]= None) -> Tensor:
        if self.include_sensitivitymaps:
            S = self.sense_matrix.to(x.get_device())
            if slice_inds is not None and slice_axis is not None:
                S = S.index_select(slice_axis+1, slice_inds) # add +1 to skip coil dim in f ront
            # x shape (1, Coils, Z, Y, X, 2)
            if self.sensitivitymaps_complex:
                x = torch.view_as_real(
                    torch.view_as_complex(x.unsqueeze(-5)) * S
                )
            else:
                x = x.unsqueeze(-5) * S

        y = fft3c(x) if not self.wrapped_2d_mode else fastmri.fft2c(x)
        if self.mask is not None:
            return y * self.mask.to(y.get_device()) + 0.0
        else:
            return y

    def trafo_adjoint(self, y: Tensor) -> Tensor:
        x_hat = ifft3c(y) if not self.wrapped_2d_mode else fastmri.ifft2c(y)

        if self.include_sensitivitymaps:
            if self.sensitivitymaps_complex:
                x_hat = torch.view_as_real(
                    torch.sum(
                        torch.conj(self.sense_matrix).to(x_hat.get_device()) * torch.view_as_complex(x_hat)
                    , dim=-4) # / self.sense_matrix_normalization_constant.to(x_hat.get_device())
                )
            else:
                x_hat = torch.sum(self.sense_matrix.to(x_hat.get_device()) * x_hat, dim=1)

        return x_hat
