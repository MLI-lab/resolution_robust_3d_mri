from typing import Tuple, List, Optional
from torch import Tensor
import torch
from .base_prior_trafo import BasePriorTrafo
from .base_target_trafo import BaseTargetTrafo
from fastmri.data.transforms import (
    complex_center_crop,
    normalize_instance,
)
import fastmri
import torch.nn.functional as F
import torchvision.transforms as T


class CroppedMagnitudeImagePriorTrafo(BasePriorTrafo, BaseTargetTrafo):
    def __init__(
        self,
        crop_size: Tuple[int, int] = (320, 320),
        center_crop_enabled: bool = False,
        magnitude_enabled: bool = False,
        normalize_enabled: bool = False,
        scaling_factor: float = 1.0,
        swap_channels: bool = False,
        move_axis: Optional[Tuple[int, int]] = None,
        stack_channel_into_batchdim: bool = False,
        squeeze_channels: Optional[List[int]] = None,
        unsqueeze_channels: Optional[List[int]] = None,
        conv_averaging_shape: Optional[Tuple[int, int, int]] = None,
        swap_spatial_channels: bool = False,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.center_crop_enabled = center_crop_enabled
        self.magnitude_enabled = magnitude_enabled
        self.normalize_enabled = normalize_enabled
        self.scaling_factor = scaling_factor
        self.swap_channels = swap_channels
        self.move_axis = move_axis
        self.squeeze_channels = squeeze_channels
        self.unsqueeze_channels = unsqueeze_channels

        self.pad_sizes = [0, 160]
        self.pad_trafo = T.Pad(self.pad_sizes)

        self.stack_channel_into_batchdim = stack_channel_into_batchdim
        self.conv_averaging_shape = (
            tuple(conv_averaging_shape) if conv_averaging_shape is not None else None
        )

        self.swap_spatial_channels = swap_spatial_channels

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection.
        this might be not correct
        Parameters
        ----------
        x : :class:`torch.Tensor`
            Image of attenuation.
        """
        if self.center_crop_enabled:
            x = complex_center_crop(x, self.crop_size)
        if self.magnitude_enabled:
            x = fastmri.complex_abs(x)
        if self.scaling_factor != 1.0:
            x = x * self.scaling_factor
        if self.normalize_enabled:
            x, _, _ = normalize_instance(x, eps=1e-11)
        if self.swap_channels:
            # (1, 256, 320, 320, 2) ->
            x = x.unsqueeze(-4).swapaxes(-4, -1).squeeze(-1)
        if self.move_axis is not None:
            x = x.moveaxis(self.move_axis[0], self.move_axis[1])
        if self.swap_spatial_channels:
            x = x.swapaxes(-2, -1)
        if self.squeeze_channels is not None:
            for i in self.squeeze_channels:
                x = x.squeeze(i)
        if self.unsqueeze_channels is not None:
            for i in self.unsqueeze_channels:
                x = x.unsqueeze(i)
        if self.stack_channel_into_batchdim:
            x = x.swapaxes(0, -4)  # this assumes bath dim=1
        if self.conv_averaging_shape is not None:
            # (1, 320, 320, 320) -> (1, target_dim, 320, 320)
            kernel = torch.ones(
                (1, 1) + self.conv_averaging_shape, device=x.get_device()
            ) / torch.prod(torch.Tensor(self.conv_averaging_shape))
            x = F.conv3d(x, weight=kernel, stride=self.conv_averaging_shape)
        return x

    def trafo_inv(self, x: Tensor) -> Tensor:
        if self.conv_averaging_shape is not None:
            for dim, rep in enumerate(self.conv_averaging_shape):
                x = x.repeat_interleave(repeats=rep, dim=dim)
        if self.stack_channel_into_batchdim:
            x = x.swapaxes(-4, 0)  # this assumes batch_dim = 1
        if self.squeeze_channels is not None:
            for i in self.squeeze_channels:
                x = x.unsqueeze(i)
        if self.unsqueeze_channels is not None:
            for i in self.unsqueeze_channels:
                x = x.squeeze(i)

        if self.swap_spatial_channels:
            x = x.swapaxes(-2, -1)
        if self.move_axis is not None:
            x = x.moveaxis(self.move_axis[1], self.move_axis[0])
        if self.swap_channels:
            x = x.unsqueeze(-1).swapaxes(-4, -1).squeeze(-4)
        if self.normalize_enabled:
            raise NotImplementedError
        if self.scaling_factor != 1.0:
            x = x / self.scaling_factor
        if self.magnitude_enabled:
            x = F.pad(x, (0, 1, 0, 0), "constant", 0)
        if self.center_crop_enabled:
            bcwh = x.unsqueeze(-4).swapaxes(-4, -1).squeeze(-1)
            bcwh_p = self.pad_trafo(bcwh)
            x = bcwh_p.unsqueeze(-1).swapaxes(-1, -4).squeeze(-4)

        return x
