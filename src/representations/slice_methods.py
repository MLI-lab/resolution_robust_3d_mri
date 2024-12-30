from omegaconf import ListConfig
import torch
from typing import Tuple, List, Optional
from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh
import math

from abc import ABC, abstractmethod


class SliceMethod(ABC):
    @abstractmethod
    def __call__(
        self, volume: torch.Tensor, outer_iteration: int, inner_iteration: int
    ) -> List[torch.Tensor]:
        pass


class RandomAverageSlabsWithSliceableMesh(SliceMethod):
    def __init__(
        self,
        slice_budget: int = 5,
        slab_thickness: int = 3,
        slice_stride: int = 0,
        volume_indices: Tuple[int, int, int] = [2, 3, 4],  # Z, Y, X
        slices_discrete: bool = False,
        random_downsample_mesh: bool = False,
        random_downsample_mesh_range: List[float] = [0.1, 1.0],
        grid_aligned: Optional[bool] = None,
        slice_enabled: Tuple[bool, bool, bool] = [True, True, True],
        average_slabs: Tuple[bool, bool, bool] = [False, False, False],
        swapaxis: Tuple[bool, bool, bool] = [False, False, False],
        rnd_indices: Tuple[bool, bool, bool] = [False, False, False],
        keep_dims: Tuple[bool, bool, bool] = [False, False, False],
    ):

        self.slice_budget = slice_budget
        self.slices_discrete = slices_discrete
        self.slab_thickness = slab_thickness
        self.grid_aligned = grid_aligned
        self.stride = slice_stride
        self.volume_indices = volume_indices
        self.slice_enabled = slice_enabled
        self.average_slabs = average_slabs
        self.swapaxis = swapaxis
        self.rnd_indices = rnd_indices
        self.keep_dims = keep_dims
        self.random_downsample_mesh = random_downsample_mesh
        self.random_downsample_mesh_range = random_downsample_mesh_range

    def __call__(
        self,
        representation: CoordBasedRepresentation,
        mesh: SliceableMesh,
        mesh_data: SliceableMesh,
        outer_iteration: int,
        inner_iteration: int,
    ) -> List[torch.Tensor]:

        slice_budget = self.slice_budget

        rep = math.ceil(float(slice_budget) / self.slab_thickness)

        ret = []
        ret_slice_inds = []
        for (
            volume_index,
            slice_enabled,
            averageing,
            swapping,
            rnd_index,
            keep_dim,
        ) in zip(
            self.volume_indices,
            self.slice_enabled,
            self.average_slabs,
            self.swapaxis,
            self.rnd_indices,
            self.keep_dims,
        ):
            if slice_enabled:
                if self.slices_discrete:

                    slice_dim = mesh.matrix_size[volume_index]

                    stride = self.stride
                    index_mask = (
                        (stride + 1)
                        * (
                            torch.arange(
                                self.slab_thickness, device=representation.device
                            )
                            - self.slab_thickness // 2
                        )
                    ).repeat(rep)
                    offset = (stride + 1) * (self.slab_thickness // 2)

                    if rnd_index:
                        slice_inds = (
                            torch.randint(
                                offset,
                                slice_dim - offset,
                                (rep,),
                                device=representation.device,
                            ).repeat_interleave(self.slab_thickness)
                            + index_mask
                        )

                        slice_inds = slice_inds[:slice_budget]
                    else:
                        assert (
                            stride == 0
                        ), "Cycling mode not yet supported with stride > 0"
                        slice_inds = (
                            (
                                offset
                                + inner_iteration * (stride + 1) * self.slab_thickness
                            )
                            * torch.ones((1,), device=representation.device)
                        ).repeat_interleave(self.slab_thickness) + index_mask
                        slice_inds = slice_inds[slice_inds < slice_dim]

                    slices = representation.forward(
                        mesh.add_index_select(volume_index, slice_inds.int())
                    )
                    ret_slice_inds.append(slice_inds)
                # else:
                    # assert (
                        # rnd_index
                    # ), "Continuous slicing only supported with random indices"

                    # if isinstance(self.stride, ListConfig):
                        # stride = self.stride[0] + torch.rand(
                            # (1,), device=representation.device
                        # ).item() * (self.stride[1] - self.stride[0])
                    # else:
                        # stride = self.stride

                    # lb, ub = (
                        # mesh.lower_coords[volume_index],
                        # mesh.upper_coords[volume_index],
                    # )  # e.g. -1.0, 1.0

                    # assert (
                        # self.grid_aligned is not None
                    # ), "grid_aligned must be set for continuous slicing"

                    # grid_aligned = self.grid_aligned

                    # stepsize = (ub - lb) / (mesh.matrix_size[volume_index] - 1)
                    # if grid_aligned:
                        # pix_start = torch.randint(
                            # 0,
                            # mesh.matrix_size[volume_index] - self.slab_thickness,
                            # (rep,),
                            # device=representation.device,
                        # )
                        # slices_lb = lb + stepsize * pix_start
                    # else:
                        # ub_red = max(
                            # ub - (self.slab_thickness - 1) * stepsize * stride, lb
                        # )
                        # slices_lb = lb + (ub_red - lb) * torch.rand(
                            # (rep,), device=representation.device
                        # )  # shape (rep,) #

                    # slices_ub = (
                        # slices_lb + (self.slab_thickness - 1) * stepsize * stride
                    # )  # shape (rep,)

                    # random_downsample_mesh = self.random_downsample_mesh
                    # random_downsample_mesh_range = self.random_downsample_mesh_range

                    # res = torch.tensor(
                        # mesh.matrix_size, device=representation.device
                    # ).float()
                    # if random_downsample_mesh:
                        # random_factor = random_downsample_mesh_range[0] + torch.rand(
                            # (1,), device=representation.device
                        # ).item() * (
                            # random_downsample_mesh_range[1]
                            # - random_downsample_mesh_range[0]
                        # )
                        # res = torch.ceil(res * random_factor).int()
                    # res[volume_index] = self.slab_thickness

                    # slices = representation.forward(
                        # mesh.add_cont_slice_select(
                            # axis=volume_index,
                            # mesh_lbs=slices_lb,
                            # mesh_ubs=slices_ub,
                            # mesh_resolutions=res,
                        # )
                    # )

                    # ret_slice_inds.append(slices_lb)

                if not keep_dim:
                    slices = slices.moveaxis(volume_index, 0)

                if averageing:
                    slices = slices.view(
                        self.slab_thickness, rep, -1, slices.shape[-2], slices.shape[-1]
                    ).mean(dim=0)
                elif not keep_dim:
                    slices = slices.squeeze(1)

                if swapping:
                    slices = slices.swapaxes(-2, -3)

                ret.append(slices)

        return ret, ret_slice_inds