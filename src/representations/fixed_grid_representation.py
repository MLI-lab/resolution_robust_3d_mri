from typing import Optional, Any, Union, Tuple

import torch
import torch.nn as nn

from torch import Tensor

from src.representations.mesh import SliceableMesh, SliceableMesh
from src.representations.base_coord_based_representation import CoordBasedRepresentation


class FixedGridRepresentation(CoordBasedRepresentation, nn.Module):

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_features: int = 1,
        warm_start: Optional[Tensor] = None,
        device: Optional[Any] = None,
    ):
        super().__init__()

        if out_features == 1:
            rndn_shape = in_shape
        else:
            rndn_shape = (*in_shape, out_features)

        self.param = nn.Parameter(
            (
                torch.randn(rndn_shape, device=device)
                if warm_start is None
                else warm_start
            ),
            requires_grad=True,
        )

        self.device = device

    def get_optimizer_params(self) -> Tuple:
        return self.parameters()

    def forward(self, mesh: Optional[SliceableMesh]) -> Tensor:
        assert mesh is None or isinstance(
            mesh, SliceableMesh
        ), "IdentityNN only supports sliceable meshes"
        if mesh is None:
            return self.param
        else:
            return mesh.apply_slicing_to_tensor(self.param)

    def forward_splitted(
        self, mesh: SliceableMesh, custom_device: Optional[Any] = None, split: int = 1
    ) -> Tensor:
        param = (
            self.param
            if (custom_device is None or custom_device == self.device)
            else self.param.cpu().to(custom_device)
        )
        return mesh.apply_slicing_to_tensor(param)