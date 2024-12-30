from typing import Tuple, Optional, Any, List

import torch
from torch import Tensor

def construct_meshgrid(
    matrix_size: List[int],
    field_of_view: List[int],
    max_coord: float = 1.0,
    centered: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    assert (
        len(matrix_size) > 1
        and len(matrix_size) < 4
        and len(matrix_size) == len(field_of_view)
    )

    max_fov = max(field_of_view)
    scaled_fov = [max_coord * fov / max_fov for fov in field_of_view]

    if centered:
        coords = [
            torch.linspace(-fov_i, fov_i, steps=matrix_size_i)
            for fov_i, matrix_size_i in zip(scaled_fov, matrix_size)
        ]
    else:
        coords = [
            torch.linspace(0, fov_i, steps=matrix_size_i)
            for fov_i, matrix_size_i in zip(scaled_fov, matrix_size)
        ]

    stepsizes = torch.tensor([coords[1] - coords[0] for coords in coords])
    lower_coords = torch.tensor([coords[0] for coords in coords])
    upper_coords = torch.tensor([coords[-1] for coords in coords])

    mgrid = torch.stack(torch.meshgrid(*coords), dim=-1)

    return mgrid, stepsizes, lower_coords, upper_coords


class SliceableMesh:

    def __init__(
        self,
        matrix_size: List[int],
        field_of_view: List[int],
        device: Optional[Any] = None,
        max_coord: float = 1.0,
        requires_coords: bool = False,
        index_selects: List[Tuple[int, Tensor]] = [],
        coord_mesh: Optional[Tensor] = None,
        coord_stepsizes: Optional[Tensor] = None,
        lower_coords: Optional[Tensor] = None,
        upper_coords: Optional[Tensor] = None,
        mesh_jitter_enable: bool = False,
        mesh_jitter_is_int: bool = True,
        mesh_jitter_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.matrix_size = matrix_size
        self.field_of_view = field_of_view
        self.max_coord = max_coord

        self.index_selects = index_selects
        self.coord_mesh = coord_mesh
        self.requires_coords = requires_coords
        self.coord_stepsizes = coord_stepsizes
        self.lower_coords = lower_coords
        self.upper_coords = upper_coords

        self.mesh_jitter_enable = mesh_jitter_enable
        self.mesh_jitter_is_int = mesh_jitter_is_int
        self.mesh_jitter_bounds = mesh_jitter_bounds

        if requires_coords:
            if self.coord_mesh is None:
                (
                    self.coord_mesh,
                    self.coord_stepsizes,
                    self.lower_coords,
                    self.upper_coords,
                ) = construct_meshgrid(
                    self.matrix_size, self.field_of_view, self.max_coord
                )
                self.coord_mesh = self.coord_mesh.to(self.device)
                self.coord_stepsizes = self.coord_stepsizes.to(self.device)
                self.lower_coords = self.lower_coords.to(self.device)
                self.upper_coords = self.upper_coords.to(self.device)

    def add_index_select(self, axis: int, indices: Tensor) -> Any:
        return SliceableMesh(
            matrix_size=self.matrix_size,
            field_of_view=self.field_of_view,
            device=self.device,
            max_coord=self.max_coord,
            requires_coords=self.requires_coords,
            index_selects=self.index_selects + [(axis, indices)],
            coord_mesh=self.coord_mesh,
            coord_stepsizes=self.coord_stepsizes,
            lower_coords=self.lower_coords,
            upper_coords=self.upper_coords,
            mesh_jitter_bounds=self.mesh_jitter_bounds,
            mesh_jitter_enable=self.mesh_jitter_enable,
            mesh_jitter_is_int=self.mesh_jitter_is_int,
        )

    def get_coord_mesh(self) -> Tensor:
        sliced_coords = self.apply_slicing_to_tensor(self.coord_mesh)

        if self.mesh_jitter_enable:
            _, _, _, C = sliced_coords.shape

            base_shape = [1, 1, 1, C]
            if self.index_selects is not None:
                base_shape[self.index_selects[0][0]] = sliced_coords.shape[
                    self.index_selects[0][0]
                ]

            mesh_jitter = (
                torch.ones(base_shape, device=self.device) * self.coord_stepsizes // 2
            )
            return sliced_coords + mesh_jitter
        else:
            return sliced_coords

    def get_rescaled_coord_mesh(self, lb: float = -1.0, ub: float = 1.0) -> Tensor:
        assert self.lower_coords is not None and self.upper_coords is not None, "lower and upper bounds not set yet"

        mesh = self.get_coord_mesh()
        lc = self.lower_coords
        uc = self.upper_coords
        return lb + (ub - lb) * (mesh - lc) / (uc - lc)[None, None, None, :]

    def apply_slicing_to_tensor(self, tensor) -> Tensor:
        for axis, indices in self.index_selects:
            tensor = tensor.index_select(axis, indices.to(tensor.device))
        return tensor
