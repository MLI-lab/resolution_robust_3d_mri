#pragma once

#include <torch/torch.h>

namespace RASTERIZER_PER_TILE {

    struct GeometryState
	{
		size_t scan_size;
		char* scanning_space;
        int3* tile_bounds_l;
        int3* tile_bounds_u;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

    struct VolumeState
    {
		uint2* ranges;
		uint32_t* n_contrib;

		static VolumeState fromChunk(char*& chunk, size_t N);
    };

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

}
