from typing import Optional, Any, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from src.representations.mesh import SliceableMesh, SliceableMesh
from src.representations.base_coord_based_representation import CoordBasedRepresentation

import logging


class GridResamplingRepresentation(CoordBasedRepresentation, nn.Module):

    def __init__(
        self,
        rep_mesh: SliceableMesh,
        out_features: int = 1,
        interpolation_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
        warm_start: Optional[Tensor] = None,
        warm_start_mesh: Optional[SliceableMesh] = None,
        device: Optional[Any] = None,
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.device = device

        in_shape = rep_mesh.matrix_size
        if out_features == 1:
            rndn_shape = in_shape
        else:
            rndn_shape = (*in_shape, out_features)

        if warm_start is None:
            self.param = nn.Parameter(
                torch.randn(rndn_shape, device=device), requires_grad=True
            )
        elif warm_start is not None and warm_start_mesh.matrix_size == in_shape:
            self.param = nn.Parameter(warm_start, requires_grad=True)
        else:
            logging.info("Upsampling the warmstart to match the rep_shape.")
            grid = rep_mesh.get_rescaled_coord_mesh(lb=-1.0, ub=1.0).to(
                self.device
            )
            grid = grid.unsqueeze(0)
            grid_reformed = torch.stack(
                [grid[:, :, :, :, 2], grid[:, :, :, :, 1], grid[:, :, :, :, 0]], dim=-1
            )
            warm_start_interp = (
                torch.nn.functional.grid_sample(
                    warm_start[None].moveaxis(-1, 1),
                    grid=grid_reformed,
                    mode=self.interpolation_mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )
                .squeeze(0)
                .moveaxis(0, -1)
                .contiguous()
            )
            self.param = nn.Parameter(warm_start_interp, requires_grad=True)

    def get_optimizer_params(self) -> Tuple:
        return self.parameters()

    def forward(self, mesh: SliceableMesh) -> Tensor:

        grid = mesh.get_rescaled_coord_mesh(lb=-1.0, ub=1.0).to(
            self.device
        )

        grid = grid.unsqueeze(0)
        grid_reformed = torch.stack(
            [grid[:, :, :, :, 2], grid[:, :, :, :, 1], grid[:, :, :, :, 0]], dim=-1
        )
        return (
            torch.nn.functional.grid_sample(
                self.param[None].moveaxis(-1, 1),
                grid=grid_reformed,
                mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            .squeeze(0)
            .moveaxis(0, -1)
            .contiguous()
        )

    def forward_splitted(
        self, mesh: SliceableMesh, custom_device: Optional[Any] = None, split: int = 1
    ) -> Tensor:

        grid = mesh.get_rescaled_coord_mesh(lb=-1.0, ub=1.0).to(self.device)
        param = (
            self.param
            if (custom_device is None or custom_device == self.device)
            else self.param.cpu().to(custom_device)
        )

        grid = grid.unsqueeze(0)
        grid_reformed = torch.stack(
            [grid[:, :, :, :, 2], grid[:, :, :, :, 1], grid[:, :, :, :, 0]], dim=-1
        )
        return (
            torch.nn.functional.grid_sample(
                param.unsqueeze(0).moveaxis(-1, 1).contiguous(),
                grid=grid_reformed,
                mode=self.interpolation_mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            .squeeze(0)
            .moveaxis(0, -1)
            .contiguous()
        )
