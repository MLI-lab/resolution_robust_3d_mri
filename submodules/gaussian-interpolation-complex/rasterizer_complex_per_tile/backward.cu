#include <iostream>
//#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "config.h"

#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "backward.h"

//template<int C>

__global__ void rasterize_basic_complex_backward_CUDA(
    const uint2* tile_ranges,
    const uint32_t* gaussian_ptr_list,
	const glm::vec3* means,
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
	float* grad_phases_add)
{
	auto idx = cg::this_grid().thread_rank();
	auto idx_in_block = cg::this_thread_block().thread_rank();

	__shared__ glm::vec3 mesh_lb_shared;
	__shared__ glm::vec3 mesh_ub_shared;
	__shared__ glm::vec3 mesh_resolutions_shared;
	__shared__ glm::vec3 voxel_offset_factors_shared;

	if (idx_in_block == 0)
		mesh_lb_shared = *mesh_lb;
	if (idx_in_block == 1)
		mesh_ub_shared = *mesh_ub;
	if (idx_in_block == 2)
		mesh_resolutions_shared = *mesh_resolutions;
	if (idx_in_block == 3)
		voxel_offset_factors_shared = *voxel_offset_factors;

	__syncthreads();

	const uint32_t idx_mapped = idx;

    // get dimensions
	const int X = (int)mesh_resolutions_shared.x; // 256
	const int Y = (int)mesh_resolutions_shared.y; // 320
	const int Z = (int)mesh_resolutions_shared.z; // 320

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

    // number of blocks (tiles) per dimension
	uint32_t blocks_x = (X + BLOCK_X - 1) / BLOCK_X;
	uint32_t blocks_y = (Y + BLOCK_Y - 1) / BLOCK_Y;
	uint32_t blocks_z = (Z + BLOCK_Z - 1) / BLOCK_Z;

    // pixel mins and pixel maxs of current tile
	uint3 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y, block.group_index().z * BLOCK_Z};
	uint3 pix_max = { min(pix_min.x + BLOCK_X, X), min(pix_min.y + BLOCK_Y , Y), min(pix_min.z + BLOCK_Z, Z)};
	uint3 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y, pix_min.z + block.thread_index().z};

    // index of pixel in full voxel
	//uint32_t pix_id = pix.z * X*Y + pix.y * X + pix.x;
	uint32_t pix_id = pix.x * Z*Y + pix.y * Z + pix.z;
    //uint32_t tile_id = block.group_index().z * blocks_x * blocks_y + block.group_index().y * blocks_x + block.group_index().x;
    uint32_t tile_id = block.group_index().x * blocks_y * blocks_z + block.group_index().y * blocks_z + block.group_index().z;

    // extract ranges of gaussians
	uint2 range = tile_ranges[tile_id];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < X && pix.y < Y && pix.z < Z && pix.x >= 0 && pix.y >= 0 && pix.z >= 0;

    // calculate pixel - coord representation
	const float stepsize_x = (mesh_ub_shared.x - mesh_lb_shared.x) / (X - 1);
	const float stepsize_y = (mesh_ub_shared.y - mesh_lb_shared.y) / (Y - 1);
	const float stepsize_z = (mesh_ub_shared.z - mesh_lb_shared.z) / (Z - 1);

	const float intra_voxel_offset_x = stepsize_x * voxel_offset_factors_shared.x;
	const float intra_voxel_offset_y = stepsize_y * voxel_offset_factors_shared.y;
	const float intra_voxel_offset_z = stepsize_z * voxel_offset_factors_shared.z;

    // evaluate voxels in center
    glm::vec3 coord = {
        mesh_lb_shared.x + pix.x * stepsize_x + intra_voxel_offset_x,
        mesh_lb_shared.y + pix.y * stepsize_y + intra_voxel_offset_y,
        mesh_lb_shared.z + pix.z * stepsize_z + intra_voxel_offset_z
    };

    glm::vec2 grad_at_index;

    if(inside) {
        grad_at_index = grad_results[pix_id];
    }

	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

    // from here everything is specific to idx_mapped gaussian
	__shared__ int       collected_id[BLOCK_SIZE];
	__shared__ glm::vec3 collected_mean[BLOCK_SIZE];
	__shared__ float     collected_opacity[BLOCK_SIZE];
    __shared__ glm::vec3 collected_scaling[BLOCK_SIZE]; // element-wise inverse of scale
    __shared__ glm::vec4 collected_rotation[BLOCK_SIZE]; // quaternion
    //__shared__ glm::mat3 collected_R[BLOCK_SIZE]; // assembled rotation matrix
	__shared__ glm::vec3 collected_phases[BLOCK_SIZE];
	__shared__ float     collected_phases_add[BLOCK_SIZE];

	// Initialize helper variables
	//float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;

    float result_re = 0.0f;
    float result_im = 0.0f;

    // temp vectors
    glm::vec3 coord_rot;
    glm::vec3 a;
    glm::vec3 scale;
    glm::vec4 rotation;

	//float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = gaussian_ptr_list[range.x + progress];

			collected_id[block.thread_rank()] = coll_id;
            collected_mean[block.thread_rank()] = means[coll_id];
            collected_opacity[block.thread_rank()] = opacities[coll_id];

            scale = scales[coll_id];
            collected_scaling[block.thread_rank()] = glm::vec3(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);

            collected_rotation[block.thread_rank()] = rotations[coll_id];
            collected_phases_add[block.thread_rank()] = phases_add[coll_id];

            if(!use_phase_add_as_imag) {
                collected_phases[block.thread_rank()] = phases[coll_id];
            }

		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

            // copy into local memory, since used multiple times
			const int global_id = collected_id[j];
            glm::vec3 scaling = collected_scaling[j];
            glm::vec3 mean = collected_mean[j];

            glm::vec4 rotation = rotations[j];
	        const float r = rotation.x;
	        const float x = rotation.y;
	        const float y = rotation.z;
	        const float z = rotation.w;

            float opacity = collected_opacity[j];
            float phase_add = collected_phases_add[j];

	        glm::mat3 R = glm::mat3(
		        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	        );

            coord_rot = R * (coord - mean);
            a = coord_rot * scaling;

            float a_norm_sq = glm::dot(a,a);
            if(a_norm_sq < scale_multiplier * scale_multiplier)
            {

                float exp_value_at_coord_mag = glm::exp(-0.5f * a_norm_sq);

                if (use_phase_add_as_imag) {

                    glm::vec2 cs = glm::vec2(opacity, phase_add);
                    //glm::vec2 ms_c = glm::vec2(-s, c); // i * cs

                    float grad_cs = glm::dot(grad_at_index, cs);
                    //float grad_ms_c = glm::dot(grad_at_index, ms_c);

                    // opacity gradient (real part)
                    atomicAdd(
                        grad_opacities+global_id,
                        exp_value_at_coord_mag * grad_at_index.x
                    );
                    // phase add gradient (imag part)
                    atomicAdd(
                        grad_phases_add+global_id,
                        exp_value_at_coord_mag * grad_at_index.y
                    );

                    // scale gradient (same as before but different gradient)
                    glm::vec3 gs = exp_value_at_coord_mag * grad_cs * a * a * scaling;
                    atomicAdd(&grad_scales[global_id].x, gs.x);
                    atomicAdd(&grad_scales[global_id].y, gs.y);
                    atomicAdd(&grad_scales[global_id].z, gs.z);

                    // mean gradient: f(x) ( Sigma * (x - mean) - i p )
                    glm::vec3 gm = exp_value_at_coord_mag * grad_cs * glm::transpose(R) * (a * scaling);
                    atomicAdd(&grad_means[global_id].x, gm.x);
                    atomicAdd(&grad_means[global_id].y, gm.y);
                    atomicAdd(&grad_means[global_id].z, gm.z);

                    // rotation gradient
                    glm::mat3 dL_dMt = - exp_value_at_coord_mag * grad_cs * glm::outerProduct(coord, scaling * a);

	                const float r = collected_rotation[j].x;
	                const float x = collected_rotation[j].y;
	                const float y = collected_rotation[j].z;
	                const float z = collected_rotation[j].w;

                    atomicAdd(&grad_rotations[global_id].x, 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]));
                    atomicAdd(&grad_rotations[global_id].y, 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]));
                    atomicAdd(&grad_rotations[global_id].z,  2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]));
                    atomicAdd(&grad_rotations[global_id].w, 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]));

                } else {


                    float value_at_coord_mag = collected_opacity[j] * exp_value_at_coord_mag;
                    glm::vec3 phase = collected_phases[j];
                    
                    float p = glm::dot(phase, coord-mean) + collected_phases_add[j]; // using coord here, but need to experiment
                    float c = glm::cos(p);
                    float s = glm::sin(p);

                    // temp
                    glm::vec2 cs = glm::vec2(c, s);
                    glm::vec2 ms_c = glm::vec2(-s, c); // i * cs

                    float grad_cs = glm::dot(grad_at_index, cs);
                    float grad_ms_c = glm::dot(grad_at_index, ms_c);

                    // opacity gradient(s)
                    atomicAdd(
                        grad_opacities+global_id,
                        exp_value_at_coord_mag * grad_cs
                    );

                    // scale gradient
                    glm::vec3 gs = value_at_coord_mag * grad_cs * a * a * scaling;
                    atomicAdd(&grad_scales[global_id].x, gs.x);
                    atomicAdd(&grad_scales[global_id].y, gs.y);
                    atomicAdd(&grad_scales[global_id].z, gs.z);
				
                    // mean gradient: f(x) ( Sigma * (x - mean) - i p )
                    glm::vec3 gm = value_at_coord_mag * ( grad_cs * glm::transpose(R) * (a * scaling) - grad_ms_c * phase);
                    atomicAdd(&grad_means[global_id].x, gm.x);
                    atomicAdd(&grad_means[global_id].y, gm.y);
                    atomicAdd(&grad_means[global_id].z, gm.z);

                    // rotation gradient
                    glm::mat3 dL_dMt = - value_at_coord_mag * grad_cs * glm::outerProduct(coord, scaling * a);

	                const float r = collected_rotation[j].x;
	                const float x = collected_rotation[j].y;
	                const float y = collected_rotation[j].z;
	                const float z = collected_rotation[j].w;

                    atomicAdd(&grad_rotations[global_id].x, 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]));
                    atomicAdd(&grad_rotations[global_id].y, 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]));
                    atomicAdd(&grad_rotations[global_id].z,  2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]));
                    atomicAdd(&grad_rotations[global_id].w, 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]));

                    // phase gradient
                    glm::vec3 gp = coord * grad_ms_c * value_at_coord_mag;
                    atomicAdd(&grad_phases[global_id].x, gp.x);
                    atomicAdd(&grad_phases[global_id].y, gp.y);
                    atomicAdd(&grad_phases[global_id].z, gp.z);

                    // phase_add gradient
                    atomicAdd(&grad_phases_add[global_id], grad_ms_c * value_at_coord_mag);

                }
            }
   
			last_contributor = contributor;
		}
	}
}

void RASTERIZER_PER_TILE::BACKWARD::rasterize_basic_complex_backward(
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
    const glm::vec2* grad_results, // 3D array of results
    glm::vec3* grad_means,
    float* grad_opacities,
    glm::vec3* grad_scales,
    glm::vec4* grad_rotations,
    glm::vec3* grad_phases,
    float* grad_phases_add) {
	rasterize_basic_complex_backward_CUDA<< < tile_grid, block >> > (
        tile_ranges, gaussian_ptr_list,
        mean, opacities, scales, rotations, phases, phases_add,
        mesh_lb, mesh_ub, mesh_resolutions, voxel_offset_factors,
        use_phase_add_as_imag, scale_multiplier, grad_padding_factor, grad_padding_const, grad_results, grad_means, grad_opacities, grad_scales, grad_rotations, grad_phases, grad_phases_add);
}