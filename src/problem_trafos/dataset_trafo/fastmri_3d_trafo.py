from typing import Tuple, List
import torch
import numpy as np
import logging
import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch

from fastmri.data.transforms import to_tensor, normalize_instance
from src.problem_trafos.dataset_trafo.mask_utils import apply_mask
import fastmri
from sigpy.mri.samp import poisson
from src.utils.fftn3d import fft3c, ifft3c

from .base_dataset_trafo import BaseDatasetTrafo
from src.datasets.fastmri_volume_dataset import VolumeDatasetSample

class FastMRI3DDataTransform(BaseDatasetTrafo[VolumeDatasetSample]):
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
        target_interpolate_by_factor : float = 1.0,
        target_interpolation_method : str = "nearest",
        normalize_target : bool = False,
        target_type : str = "rss",
        pseudoinverse_conv_averaging_shape : Optional[Tuple[int, int]] = None,
        multicoil_reduction_op : bool = "sum",
        device : str = "cpu",
        wrapped_2d : bool = False,
        return_pseudoinverse_as_observation : bool = False,
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

        self.scale_target_by_kspacenorm = scale_target_by_kspacenorm
        self.target_scaling_factor = target_scaling_factor
        self.target_interpolate_by_factor = target_interpolate_by_factor
        self.target_interpolation_method = target_interpolation_method

        self.pseudoinverse_conv_averaging_shape = pseudoinverse_conv_averaging_shape
        self.wrapped_2d = wrapped_2d
    
        self.device = device
    
        self.multicoil_reduction_op = multicoil_reduction_op

        self.return_pseudoinverse_as_observation = return_pseudoinverse_as_observation

        self.mask_accelerations = mask_accelerations
        self.mask_seed = mask_seed 

        self.mask_type = mask_type
        self.mask_enabled = mask_enabled
        if mask_enabled:
            if mask_type == 'Poisson2D':
                pass
            elif mask_type == "Gaussian2D":
                pass
            else:
                mask_class = fastmri.data.subsample.RandomMaskFunc if mask_type == 'random' else fastmri.data.subsample.EquispacedMaskFractionFunc
                self.mask_func = mask_class(center_fractions=mask_center_fractions, accelerations=[mask_accelerations], seed=mask_seed)
                self.seed = mask_seed
        else:
            self.mask_func = None

    def requires_sensmaps(self) -> bool:
        return ((self.provide_pseudoinverse or self.target_type == "fullysampled_rec") and self.multicoil_reduction_op == "norm_sum_sensmaps") or self.provide_measurement

    def _transform(
        self,
        sample: VolumeDatasetSample
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:


        kspace, target, attrs = sample.kspace, sample.target, sample.attrs
        masked_kspace, target_torch, image = None, None, None

        crop_size = (320, 320)
        attrs["mask"] = None

        kspace_torch = to_tensor(kspace) if not torch.is_tensor(kspace) else kspace

        if self.device is not None:
            kspace_torch = kspace_torch.to(self.device)

        # first the target
        if target is not None:

            if self.target_type == "rss":
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target

                if self.device is not None:
                    target_torch = target_torch.to(self.device)

            elif self.target_type == "mvue":

                target_torch = to_tensor(target) if not torch.is_tensor(target) else target

                if self.device is not None:
                    target_torch = target_torch.to(self.device)

                target_torch = torch.view_as_real(target_torch)

            elif self.target_type == "fullysampled_rec":
                if not self.wrapped_2d:
                    target_torch = ifft3c(kspace_torch)
                else:
                    target_torch = fastmri.ifft2c(kspace_torch)

                if self.which_challenge == "multicoil":
                    if self.multicoil_reduction_op == "sum":
                        target_torch = target_torch.sum(dim=0) # ??
                    elif self.multicoil_reduction_op == "mean":
                        target_torch = target_torch.mean(dim=0)
                    elif self.multicoil_reduction_op == "norm":
                        target_torch = target_torch.norm(dim=0)
                    elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                        S = torch.from_numpy(attrs["sens_maps"]).movedim(-1,0).to(self.device)
                        target_torch = torch.view_as_real(torch.sum(torch.view_as_complex(target_torch) * torch.conj(S), dim=0))
                    else:
                        raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

            if self.scale_target_by_kspacenorm:
                target_torch =  target_torch * math.sqrt(float(np.prod(target_torch.shape).item())) / attrs["kspace_vol_norm"] #kspace_torch.norm()
            
            if self.target_scaling_factor != 1.0:
                target_torch = target_torch * self.target_scaling_factor

            if self.target_interpolate_by_factor != 1.0:
                target_torch = torch.nn.functional.interpolate(target_torch.movedim(-1, 0).unsqueeze(0), scale_factor=self.target_interpolate_by_factor, mode=self.target_interpolation_method).squeeze(0).movedim(0,-1).contiguous()

            if self.normalize_target:
                target_torch, mean, std = normalize_instance(target_torch, eps=1e-11)
                target_torch = target_torch.clamp(-6, 6)

        else:
            target_torch = torch.Tensor([0])

        if self.provide_pseudoinverse or self.provide_measurement:

            if self.use_real_synth_data:

                assert self.wrapped_2d == False, "Wrapped 2D not supported for real synth data"
                if self.target_type == "rss":
                    kspace_torch = fft3c(torch.stack([target_torch, torch.zeros_like(target_torch)], dim=-1))
                elif self.target_type == "fullysampled_rec":    
                    kspace_torch = fft3c(target_torch) # use sensitivity matrix?
                else:
                    raise NotImplementedError(f"Target type {self.target_type} not supported")

            if self.target_interpolate_by_factor != 1.0:

                assert not self.wrapped_2d, "Wrapped 2D not supported for interpolating target"
                logging.info(f"Target interpolated by factor {self.target_interpolate_by_factor} and requires measurement or pseudoinverse -> interpolate sensitivities and kspace")

                S_reform = torch.view_as_real(torch.from_numpy(attrs["sens_maps"])).permute(-2, -1, 0, 1, 2).to(self.device)
                S_interpolated = torch.nn.functional.interpolate(
                    S_reform,
                    scale_factor=self.target_interpolate_by_factor,
                    mode=self.target_interpolation_method).moveaxis(1, -1).contiguous() # (Coils, Z', Y', X', 2)
                S_interpolated = torch.view_as_complex(S_interpolated)
                S_new = S_interpolated.moveaxis(0,-1).contiguous()
                attrs["sens_maps"] = S_new.cpu().numpy()
                logging.info(f"sense_map: {self.device}")

                x_sens = torch.view_as_real(
                    torch.view_as_complex(target_torch.unsqueeze(0)) * S_interpolated
                )
                kspace_torch = fft3c(x_sens)

            if self.mask_enabled:
                if self.mask_type == 'Poisson2D' or self.mask_type == "Gaussian2D":

                    ms1 = int(kspace.shape[-3] * self.target_interpolate_by_factor)
                    ms2 = int(kspace.shape[-2] * self.target_interpolate_by_factor)

                    if self.mask_type == "Poisson2D":
                        self.mask = torch.from_numpy(poisson([ms1, ms2], self.mask_accelerations, seed=self.mask_seed).astype(np.float32)).unsqueeze(dim=-1)
                    else:
                        from src.problem_trafos.dataset_trafo.mask_utils import get_gaussian_2d_mask_rej
                        self.mask = get_gaussian_2d_mask_rej([ms1, ms2], acc_factor = self.mask_accelerations, seed=self.mask_seed).unsqueeze(-1)

                    self.mask_func = None

                    if self.device != "cpu":
                        self.mask = self.mask.to(self.device)

                    masked_kspace =  kspace_torch * self.mask.to(kspace_torch.device) + 0.0

                    attrs["mask"] = self.mask
                else:
                    masked_kspace, _, _ = apply_mask(kspace_torch, self.mask_func, seed=self.seed)
            else:
                masked_kspace = kspace_torch

        if self.provide_pseudoinverse:
            if not self.wrapped_2d:
                image = ifft3c(masked_kspace)
            else:
                image = fastmri.ifft2c(masked_kspace)

            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            if self.return_cropped_pseudoinverse:
                raise NotImplementedError("Cropping not supported for 3D data")

            if self.pseudoinverse_conv_averaging_shape is not None:
                raise NotImplementedError("Convolutional averaging not supported for 3D data")

            if self.return_magnitude_image:
                # absolute value
                image = fastmri.complex_abs(image)

                if self.which_challenge == "multicoil":
                    image = fastmri.rss(image)

                image = image.unsqueeze(-1)
            elif self.which_challenge == "multicoil" and not self.use_real_synth_data: 
                if self.multicoil_reduction_op == "sum":
                    image = image.sum(dim=0)
                elif self.multicoil_reduction_op == "mean":
                    image = image.mean(dim=0)
                elif self.multicoil_reduction_op == "norm":
                    image = image.norm(dim=0)
                elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                    S = torch.from_numpy(attrs["sens_maps"]).movedim(-1,0).to(self.device)
                    image = torch.view_as_real(torch.sum(torch.view_as_complex(image) * torch.conj(S), dim=0))
                else:
                    raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

        if self.device != "cpu":
            torch.cuda.empty_cache()

        if self.provide_measurement and self.return_pseudoinverse_as_observation:
            assert self.provide_pseudoinverse, "Pseudoinverse must be provided if it is returned as observation."
            masked_kspace = image.clone()

        return masked_kspace, target_torch, image, attrs