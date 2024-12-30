

#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace RASTERIZER_PER_TILE {
	namespace BACKWARD
	{
		void rasterize_basic_complex_backward(
        	dim3 tile_grid, dim3 block,
            const uint2* tile_ranges,
            const uint32_t* gaussian_ptr_list,
	    	const glm::vec3* mean,
			const float* opacities,
	    	const glm::vec3* scales,
	    	const glm::vec4* rotations,
			const glm::vec3* phases,
			const float* phases_add,
			const glm::vec3* mesh_lb, // e.g. (-1, -1, -1)
			const glm::vec3* mesh_ub, // e.g. (1, 1, 1)
			const glm::vec3* mesh_resolutions, // e.g. (128, 128, 128)
    		const glm::vec3* voxel_offset_factors,
			const bool use_phase_add_as_imag,
			const float scale_multiplier,
			const float grad_padding_factor,
			const float grad_padding_const,
        	const glm::vec2* grad_results, 
        	glm::vec3* grad_means,
        	float* grad_opacities,
        	glm::vec3* grad_scales,
        	glm::vec4* grad_rotations,
			glm::vec3* grad_phases,
			float* grad_phases_add
    	);
	}
}