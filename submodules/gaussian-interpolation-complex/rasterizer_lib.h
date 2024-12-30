
#pragma once

#include <torch/torch.h>

void rasterize_gaussians_cuda_complex(
    const torch::Tensor& means,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& phases,
    const torch::Tensor& phases_add,
    const torch::Tensor& mesh_lb,
    const torch::Tensor& mesh_ub,
    const torch::Tensor& mesh_resolutions,
    const torch::Tensor& voxel_offset_factors,
    torch::Tensor& results,
    const bool use_phase_add_as_imag,
    const float scale_multiplier,
    const int nr_gaussians
);

void rasterize_gaussians_cuda_complex_backward(
    const torch::Tensor& means,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& phases,
    const torch::Tensor& phases_add,
    const torch::Tensor& mesh_lb,
    const torch::Tensor& mesh_ub,
    const torch::Tensor& mesh_resolutions,
    const torch::Tensor& voxel_offset_factors,
	const float grad_padding_factor,
	const float grad_padding_const,
    const torch::Tensor& grad_results,
    torch::Tensor& grad_means,
    torch::Tensor& grad_opacities,
    torch::Tensor& grad_scales,
    torch::Tensor& grad_rotations,
    torch::Tensor& grad_phases,
    torch::Tensor& grad_phases_add,
    const bool use_phase_add_as_imag,
    const float scale_multiplier,
    const int nr_gaussians
);