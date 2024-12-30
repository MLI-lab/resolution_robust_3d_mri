from typing import Tuple, Union, Iterator, List
from itertools import repeat

import torch
import numpy as np

from pathlib import Path
from torch import Tensor
from PIL import Image
import logging

import xml.etree.ElementTree as etree
from copy import deepcopy
import math
from pathlib import Path
from torch.utils.data import ConcatDataset
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import numpy as np
import pandas as pd
import torch

from fastmri.data.transforms import MaskFunc, to_tensor, complex_center_crop, normalize_instance
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.dataset_trafo.mask_utils import apply_mask
import fastmri

from src.datasets.fastmri_volume_dataset import FastMRIVolumeDataset
from fastmri.data.mri_data import FastMRIRawDataSample

from sigpy.mri.samp import poisson, radial, spiral
import torchvision.transforms.functional as TF

from .base_dataset_trafo import BaseDatasetTrafo
from src.datasets.fastmri_slice_dataset import SliceDatasetSample

class FastMRI2DDataTransform(BaseDatasetTrafo[SliceDatasetSample]):
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_enabled : bool, mask_type : str, mask_accelerations : Tuple[int],
        mask_center_fractions : Tuple[float], mask_seed : int,
        use_seed: bool = True,
        provide_pseudoinverse : bool = False,
        provide_measurement : bool = True,
        use_real_synth_data : bool = False,
        return_magnitude_image : bool = False,
        return_cropped_pseudoinverse : bool = False,
        scale_target_by_kspacenorm : bool = False,
        target_scaling_factor : float = 1.0,
        target_random_crop_size : Optional[Tuple[int, int]] = None,
        normalize_target : bool = False,
        target_type : str = "rss",
        pseudoinverse_conv_averaging_shape : Optional[Tuple[int, int]] = None,
        multicoil_reduction_op : bool = "sum",
        target_interpolate_by_factor : float = 1.0,
        target_interpolate_factor_is_interval : bool = False,
        target_interpolate_method : str = "nearest",
        device : str = "cpu",
        fwd_trafo : BaseFwdTrafo = None
    ):

        super().__init__(provide_measurement=provide_measurement, provide_pseudoinverse=provide_pseudoinverse)

        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.use_real_synth_data = use_real_synth_data
        self.return_magnitude_image = return_magnitude_image
        self.return_cropped_pseudoinverse = return_cropped_pseudoinverse
        self.normalize_target = normalize_target
        self.target_type = target_type
        self.target_random_crop_size = target_random_crop_size

        self.scale_target_by_kspacenorm = scale_target_by_kspacenorm
        self.target_scaling_factor = target_scaling_factor
        self.target_interpolate_by_factor = target_interpolate_by_factor
        self.target_interpolate_factor_is_interval = target_interpolate_factor_is_interval
        self.target_interpolate_method = target_interpolate_method

        self.pseudoinverse_conv_averaging_shape = pseudoinverse_conv_averaging_shape
        self.device = device
        self.multicoil_reduction_op = multicoil_reduction_op

        self.mask_type = mask_type
        self.mask_center_fractions = mask_center_fractions
        self.mask_accelerations = mask_accelerations
        self.mask_seed = mask_seed
        if mask_enabled:
            if self.mask_type == 'Poisson2D':
                pass
            else:
                mask_class = fastmri.data.subsample.RandomMaskFunc if mask_type == 'random' else fastmri.data.subsample.EquispacedMaskFractionFunc
                self.mask_func = mask_class(center_fractions=mask_center_fractions, accelerations=[mask_accelerations], seed=mask_seed)
                self.seed = mask_seed
        else:
            self.mask_func = None

    def requires_sensmaps(self):
        return (self.provide_pseudoinverse or self.provide_measurement or self.target_type == "fullysampled_rec") and self.multicoil_reduction_op == "norm_sum_sensmaps"

    def _transform(
        self,
        sample : SliceDatasetSample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:

        kspace, target, attrs = sample.kspace, sample.target, sample.attrs
        masked_kspace, target_torch, image = None, None, None
        crop_size = (320, 320)

        kspace_torch = to_tensor(kspace) if not torch.is_tensor(kspace) else kspace

        if self.device is not None:
            kspace_torch = kspace_torch.to(self.device)

        if self.provide_pseudoinverse or self.provide_measurement:
            if self.use_real_synth_data:
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target
                kspace_torch = fastmri.fft2c(torch.stack([target_torch, torch.zeros_like(target_torch)], dim=-1))

            if self.mask_func is not None:
                if self.mask_type == 'Poisson2D':
                    print(f"mask shape: {kspace_torch.shape}, acceleration: {self.mask_accelerations}")
                    self.mask = torch.from_numpy(poisson(kspace_torch.shape[-3:-1], self.mask_accelerations, seed=self.mask_seed).astype(np.float32)).unsqueeze(dim=-1)
                    masked_kspace = kspace_torch * self.mask.to(kspace_torch.get_device())
                else:
                    masked_kspace, _, _ = apply_mask(kspace_torch, self.mask_func, seed=self.seed)
            else:
                masked_kspace = kspace_torch

        if self.provide_pseudoinverse:
            image = fastmri.ifft2c(masked_kspace)

            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            if self.return_cropped_pseudoinverse:
                image = complex_center_crop(image, crop_size)

            if self.pseudoinverse_conv_averaging_shape is not None:
                for dim, rep in enumerate(self.pseudoinverse_conv_averaging_shape):
                    image = image.repeat_interleave(repeats=rep, dim=dim)

            if self.return_magnitude_image:
                image = fastmri.complex_abs(image)
                if self.which_challenge == "multicoil":
                    image = fastmri.rss(image)
                image = image.unsqueeze(-1)

            elif self.which_challenge == "multicoil":
                if self.multicoil_reduction_op == "sum":
                    image = image.sum(dim=-4)
                elif self.multicoil_reduction_op == "mean":
                    image = image.mean(dim=-4)
                elif self.multicoil_reduction_op == "norm":
                    image = image.norm(dim=-4)
                elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                    S = torch.from_numpy(attrs["sens_maps"]).to(image.device) # shape is: (Coils, X, Y)
                    image = torch.view_as_real(torch.sum(torch.view_as_complex(image) * torch.conj(S), dim=0))
                else:
                    raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

        # normalize target
        if target is not None:

            if self.target_type == "rss":
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target

                if self.device is not None:
                    target_torch = target_torch.to(self.device)

            elif self.target_type == "mvue":
                    target_torch = to_tensor(target) if not torch.is_tensor(target) else target
    
                    if self.device is not None:
                        target_torch = target_torch.to(self.device)
    
                    #target_torch = torch.view_as_real(target_torch)
                
            elif self.target_type == "fullysampled_rec":
                target_torch = fastmri.ifft2c(kspace_torch)
                if self.which_challenge == "multicoil":
                    if self.multicoil_reduction_op == "sum":
                        target_torch = target_torch.sum(dim=0) # ??
                    elif self.multicoil_reduction_op == "mean":
                        target_torch = target_torch.mean(dim=0)
                    elif self.multicoil_reduction_op == "norm":
                        target_torch = target_torch.norm(dim=0)
                    elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                        S = torch.from_numpy(attrs["sens_maps"]).to(target_torch.device)
                        target_torch = torch.view_as_real(torch.sum(torch.view_as_complex(target_torch) * torch.conj(S), dim=dim))
                    else:
                        raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

            if self.scale_target_by_kspacenorm:
                # for rss images one would need another "sqrt(2)" for the target vol shape product
                target_torch =  target_torch * math.sqrt(float(np.prod(attrs["target_vol_shape"]).item())) / attrs["kspace_vol_norm"]
            
            if self.target_scaling_factor != 1.0:
                target_torch = target_torch * self.target_scaling_factor

            if self.target_interpolate_by_factor is not None:
                
                if isinstance(self.target_interpolate_by_factor, str):
                    self.target_interpolate_by_factor = eval(self.target_interpolate_by_factor)

                if isinstance(self.target_interpolate_by_factor, float):
                    factor = self.target_interpolate_by_factor
                else:
                    if self.target_interpolate_factor_is_interval:
                        rnd_factor = torch.rand(size=(1,)).item()
                        factor = self.target_interpolate_by_factor[0] + rnd_factor * (self.target_interpolate_by_factor[1] - self.target_interpolate_by_factor[0])
                    else:
                        factor = self.target_interpolate_by_factor[torch.randint(0, len(self.target_interpolate_by_factor), size=(1,)).item()]
    
                target_torch = torch.nn.functional.interpolate(target_torch.movedim(-1, 0).unsqueeze(0), scale_factor=factor, mode=self.target_interpolate_method).squeeze(0).movedim(0,-1)

            if self.target_random_crop_size is not None:
                i = torch.randint(0, target_torch.shape[-3]-self.target_random_crop_size[0] + 1, size=(1,)).item()
                j = torch.randint(0, target_torch.shape[-2]-self.target_random_crop_size[1] + 1, size=(1,)).item()
                target_torch = TF.crop(target_torch.movedim(-1, 0).unsqueeze(0), i, j, self.target_random_crop_size[0], self.target_random_crop_size[1]).squeeze(0).movedim(0, -1)

            if self.normalize_target:
                target_torch, mean, std = normalize_instance(target_torch, eps=1e-11)
                target_torch = target_torch.clamp(-6, 6)

        else:
            target_torch = torch.Tensor([0])

        return masked_kspace, target_torch, image, attrs