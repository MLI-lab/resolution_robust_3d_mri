#include <torch/torch.h>
#include "rasterizer_complex_per_tile/rasterizer_impl.h"

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
    const int nr_gaussians) {

    RASTERIZER_PER_TILE::rasterize_gaussians_cuda_complex(
        means,
        opacities,
        scales,
        rotations,
        phases,
        phases_add,
        mesh_lb,
        mesh_ub,
        mesh_resolutions,
        voxel_offset_factors,
        results,
        use_phase_add_as_imag,
        scale_multiplier,
        nr_gaussians);
}

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
    const int nr_gaussians) {

    RASTERIZER_PER_TILE::rasterize_gaussians_cuda_complex_backward(
        means,
        opacities,
        scales,
        rotations,
        phases,
        phases_add,
        mesh_lb,
        mesh_ub,
        mesh_resolutions,
        voxel_offset_factors,
        grad_padding_factor,
        grad_padding_const,
        grad_results,
        grad_means,
        grad_opacities,
        grad_scales,
        grad_rotations,
        grad_phases,
        grad_phases_add,
        use_phase_add_as_imag,
        scale_multiplier,
        nr_gaussians);
}