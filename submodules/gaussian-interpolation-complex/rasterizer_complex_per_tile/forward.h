
#pragma once

#include <glm/glm.hpp>

namespace RASTERIZER_PER_TILE
{
	namespace FORWARD
	{
		// Main rasterization method.
        // One block per tile
        // Each of the thread handles one pixel
		void rasterize_basic_complex(
            dim3 tile_grid, dim3 block,
            const uint2* tile_ranges, // len=nr_tiles, start and end indices of pointers to gaussians
            const uint32_t* gaussian_ptr_list, // list of duplicated pointers to gaussians, sorted by tiles
	    	const glm::vec3* mean,
			const float* opacities,
	    	const glm::vec3* scales,
	    	const glm::vec4* rotations,
			const glm::vec3* phase,
			const float* phases_add,
			const glm::vec3* mesh_lb, // e.g. (-1, -1, -1)
			const glm::vec3* mesh_ub, // e.g. (1, 1, 1)
			const glm::vec3* mesh_resolutions, // e.g. (128, 128, 128)
    		const glm::vec3* voxel_offset_factors,
			const bool use_phase_add_as_imag,
			const float scale_multiplier,
        	glm::vec2* result
    	);
	}
}