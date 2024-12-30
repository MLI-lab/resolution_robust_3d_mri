
from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def rasterize_gaussians(
    means,
    opacities,
    scales,
    rotations,
    phases,
    phases_add,
    raster_settings
):
    return _RasterizeComplexGaussians.apply(
        means,
        opacities,
        scales, 
        rotations,
        phases,
        phases_add,
        raster_settings, 
    )

class _RasterizeComplexGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means,
        opacities,
        scales, 
        rotations,
        phases,
        phases_add,
        raster_settings, 
    ):

        # allocate memory for result
        results = torch.zeros( (raster_settings.mesh_resolutions[0].int().item(), raster_settings.mesh_resolutions[1].int().item(), raster_settings.mesh_resolutions[2].int().item(), 2) , device=means.device, dtype=means.dtype) 
        args = (
            means,
            opacities,
            scales,
            rotations,
            phases,
            phases_add,
            raster_settings.mesh_lb,
            raster_settings.mesh_ub,
            raster_settings.mesh_resolutions,
            raster_settings.voxel_offset_factors,
            results,
            raster_settings.use_phase_add_as_imag,
            raster_settings.scale_multiplier,
            means.shape[0] # nr of gaussians
        )

        # Invoke C++/CUDA rasterizer
        _C.rasterize(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(means, opacities, scales, rotations, phases, phases_add)

        return results

    @staticmethod
    def backward(ctx, grad_out):

        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        means, opacities, scales, rotations, phases, phases_add = ctx.saved_tensors

        # allocate memory for gradients
        grad_means = torch.zeros_like(means)
        grad_opacities = torch.zeros_like(opacities)
        grad_scales = torch.zeros_like(scales)
        grad_rotations = torch.zeros_like(rotations)
        grad_phases = torch.zeros_like(phases)
        grad_phases_add = torch.zeros_like(phases_add)

        # Restructure args as C++ method expects them
        args = (
            means,
            opacities,
            scales,
            rotations,
            phases,
            phases_add,
            raster_settings.mesh_lb,
            raster_settings.mesh_ub,
            raster_settings.mesh_resolutions,
            raster_settings.voxel_offset_factors,
            raster_settings.grad_padding_factor,
            raster_settings.grad_padding_const,
            grad_out,
            grad_means,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_phases,
            grad_phases_add,
            raster_settings.use_phase_add_as_imag,
            raster_settings.scale_multiplier,
            means.shape[0] # nr of gaussians
        )

        _C.rasterize_backward(*args)

        grads = (
            grad_means,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_phases,
            grad_phases_add,
            None, # raster settings
        )

        return grads

class ComplexGaussianRasterizationSettings(NamedTuple):
    scale_multiplier : float
    use_phase_add_as_imag : bool
    grad_padding_factor : float
    grad_padding_const : float
    mesh_lb : torch.Tensor
    mesh_ub : torch.Tensor
    mesh_resolutions : torch.Tensor
    voxel_offset_factors : torch.Tensor

class ComplexGaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means, opacities,
        scales, rotations,
        phases, phases_add,
        ):
        
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means,
            opacities,
            scales, 
            rotations,
            phases,
            phases_add,
            raster_settings, 
        )