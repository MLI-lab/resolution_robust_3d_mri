#include <iostream>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include "forward.h"
#include "backward.h"
#include "rasterizer_impl.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <functional>

#include "../aux.h"
#include "inline_aux.h"

#include <cooperative_groups.h>
#include <glm/glm.hpp>
namespace cg = cooperative_groups;


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int3* tile_bounds_l,
    int3* tile_bounds_u,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// offsets is set to pixels within cube
	if (offsets[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

        // tile bounds into local memory
        int3 rect_min = tile_bounds_l[idx];
        int3 rect_max = tile_bounds_u[idx];

        if(rect_min.x < 0 || rect_min.y < 0 || rect_min.z < 0)
            return;

        if(rect_max.x < 0 || rect_max.y < 0 || rect_max.z < 0)
            return;

        for (int z = rect_min.z; z <= rect_max.z; z++)
        {
		    for (int y = rect_min.y; y <= rect_max.y; y++)
		    {
			    for (int x = rect_min.x; x <= rect_max.x; x++)
			    {
                    //uint32_t key = z * grid.x * grid.y + y * grid.x + x;
                    uint32_t key = x * grid.z * grid.y + y * grid.z + z;
                    gaussian_keys_unsorted[off] = key;
                    gaussian_values_unsorted[off] = idx;
                    off++;
			    }
		    }
        }
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit. 2 0 1 3 -> 2 2 3 5
	uint32_t key = point_list_keys[idx];
	uint32_t currtile = key; // key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1]; // >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

RASTERIZER_PER_TILE::GeometryState RASTERIZER_PER_TILE::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	//obtain(chunk, geom.depths, P, 128);
	//obtain(chunk, geom.clamped, P * 3, 128);
	//obtain(chunk, geom.internal_radii, P, 128);
	//obtain(chunk, geom.means2D, P, 128);
	//obtain(chunk, geom.cov3D, P * 6, 128);
	//obtain(chunk, geom.conic_opacity, P, 128);
	//obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
    obtain(chunk, geom.tile_bounds_l, P, 128);
    obtain(chunk, geom.tile_bounds_u, P, 128);
	return geom;
}

RASTERIZER_PER_TILE::VolumeState RASTERIZER_PER_TILE::VolumeState::fromChunk(char*& chunk, size_t N)
{
	VolumeState vol;
	//obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, vol.n_contrib, N, 128);
	obtain(chunk, vol.ranges, N, 128);
	return vol;
}


__global__ void preprocess(int P,
	const glm::vec3* means,
	const glm::vec3* scales,
    const glm::vec3* mesh_lb,
    const glm::vec3* mesh_ub,
    const glm::vec3* mesh_resolutions,
	uint32_t* tiles_touched,
    int3* tile_bounds_l,
    int3* tile_bounds_u,
    dim3 tile_grid,
	const float scale_modifier)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	auto idx_in_block = cg::this_thread_block().thread_rank();

	__shared__ glm::vec3 mesh_lb_shared;
	__shared__ glm::vec3 mesh_ub_shared;
	__shared__ glm::vec3 mesh_resolutions_shared;

	if (idx_in_block == 0)
		mesh_lb_shared = *mesh_lb;
	if (P < 2 || idx_in_block == 1)
		mesh_ub_shared = *mesh_ub;
	if (P < 3 || idx_in_block == 2)
		mesh_resolutions_shared = *mesh_resolutions;

	__syncthreads();

    const glm::vec3 scale = scales[idx];
    const glm::vec3 mean = means[idx];

    glm::vec3 l, u;
    getTileBounds(mesh_resolutions_shared, mesh_lb_shared, mesh_ub_shared, mean, scale, scale_modifier, tile_grid, l, u);

    // rework
    //tile_bounds_l[idx] = {(int)l.x, (uint)l.y, (uint)l.z};
    //tile_bounds_u[idx] = {(int)u.x, (uint)u.y, (uint)u.z};
    //tiles_touched[idx] = (1 + (uint)u.x - (uint)l.x) * (1 + (uint)u.y - (uint)l.y) * (1 + (uint)u.z - (uint)l.z); 
    // tile_bounds_l[idx] = {
        // glm::max( (int)l.x, 0),
        // glm::max( (int)l.y, 0),
        // glm::max( (int)l.z, 0)
    // };
    // tile_bounds_u[idx] = {(int)u.x, (int)u.y, (int)u.z};

    int x_dist = 0;
    if (u.x < 0 || (l.x >= tile_grid.x)) {
        // gaussian is not visible
        x_dist = 0;
        tile_bounds_l[idx].x = -1;
        tile_bounds_u[idx].x = -1;
    } else {
        // is visible, crop to boundaries
        int lx2 = glm::max((int)l.x, 0);
        int ux2 = glm::min((int)u.x, (int)tile_grid.x-1);
        x_dist = (ux2 - lx2 + 1);
        tile_bounds_l[idx].x = lx2;
        tile_bounds_u[idx].x = ux2;
    }

    int y_dist = 0;
    if (u.y < 0 || (l.y >= tile_grid.y)) {
        // gaussian is not visible
        y_dist = 0;
        tile_bounds_l[idx].y = -1;
        tile_bounds_u[idx].y = -1;
    } else {
        // is visible, crop to boundaries
        float ly2 = glm::max((int)l.y, 0);
        float uy2 = glm::min((int)u.y, (int)tile_grid.y-1);
        y_dist = (uy2 - ly2 + 1);
        tile_bounds_l[idx].y = ly2;
        tile_bounds_u[idx].y = uy2;
    }

    int z_dist = 0;
    if (u.z < 0 || (l.z >= tile_grid.z)) {
        // gaussian is not visible
        z_dist = 0;
        tile_bounds_l[idx].z = -1;
        tile_bounds_u[idx].z = -1;
    } else {
        // is visible, crop to boundaries
        float lz2 = glm::max((int)l.z, 0);
        float uz2 = glm::min((int)u.z, (int)tile_grid.z-1);
        z_dist = (uz2 - lz2 + 1);
        tile_bounds_l[idx].z = lz2;
        tile_bounds_u[idx].z = uz2;
    }

    tiles_touched[idx] = x_dist * y_dist * z_dist;
}

void generate_sorted_lists(
    torch::Tensor geometry_buffer, torch::Tensor binning_buffer, torch::Tensor volstate_buffer,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& mesh_lb,
    const torch::Tensor& mesh_ub,
    const torch::Tensor& mesh_resolutions,
    const torch::Tensor& voxel_offset_factors,
    const float scale_multiplier,
    const int nr_gaussians,
    dim3 tile_grid,
    const torch::Tensor& results,
    uint32_t*& point_list,
    uint2*& ranges
) {
    // geometry_buffer: contains number of touched tiles, offsets + scanning space
    auto geometry_func = resizeFunctional(geometry_buffer);
    size_t chunk_size = required<RASTERIZER_PER_TILE::GeometryState>(nr_gaussians);
	char* chunkptr = geometry_func(chunk_size);
	RASTERIZER_PER_TILE::GeometryState geomState = RASTERIZER_PER_TILE::GeometryState::fromChunk(chunkptr, nr_gaussians);

    // preprocess: fill list of touched tiles and point offsets
    preprocess << <(nr_gaussians + BLOCK_SIZE-1) / BLOCK_SIZE, BLOCK_SIZE >> > (nr_gaussians,
        (glm::vec3*)means.contiguous().data_ptr<float>(),
        (glm::vec3*)scales.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_lb.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_ub.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_resolutions.contiguous().data_ptr<float>(),
        geomState.tiles_touched,
        geomState.tile_bounds_l,
        geomState.tile_bounds_u,
        tile_grid,
        scale_multiplier
    );

    //std::cout << "inclusive sum" << results.sum().item<float>() << std::endl;

    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, nr_gaussians);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + nr_gaussians - 1, sizeof(int), cudaMemcpyDeviceToHost);

    //std::cout << "num rendered" << num_rendered << std::endl;
    //std::cout << "binning buffer: " << results.sum().item<float>() << std::endl;

    // binning_buffer: contains keys and values for sorting + scanning space
    auto binning_func = resizeFunctional(binning_buffer);
    size_t bin_chunk_size = required<BinningState>(num_rendered);
	char* bin_chunkptr = binning_func(bin_chunk_size);
	BinningState binningState = BinningState::fromChunk(bin_chunkptr, num_rendered);

    //std::cout << "duplicate" << results.sum().item<float>() << std::endl;

	// iterate list of 
	duplicateWithKeys << <(nr_gaussians + BLOCK_SIZE-1) / BLOCK_SIZE, BLOCK_SIZE >> > (
		nr_gaussians,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
        geomState.tile_bounds_l,
        geomState.tile_bounds_u,
		tile_grid);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y * tile_grid.z);

    //std::cout << "sorting" << results.sum().item<float>() << std::endl;

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered), 0); //, 0, bit);

    //std::cout << "volState" << results.sum().item<float>() << std::endl;

    //const int volsize = X * Y * Z; // formely width*height
    //const int tile_count =  X * Y * Z; // formely width*height ??
    const int tile_count = tile_grid.x * tile_grid.y * tile_grid.z; // X * Y * Z; // formely width*height ??
    auto volstate_func = resizeFunctional(volstate_buffer);
    size_t vol_chunk_size = required<RASTERIZER_PER_TILE::VolumeState>(tile_count);
	char* vol_chunkptr = volstate_func(vol_chunk_size);
	RASTERIZER_PER_TILE::VolumeState volState = RASTERIZER_PER_TILE::VolumeState::fromChunk(vol_chunkptr, tile_count);

	cudaMemset(volState.ranges, 0, tile_count * sizeof(uint2));

    //std::cout << "identifying  " << results.sum().item<float>() << std::endl;

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + BLOCK_SIZE-1) / BLOCK_SIZE, BLOCK_SIZE >> > (
			num_rendered,
			binningState.point_list_keys,
			volState.ranges);

    //std::cout << "rasterizing " << results.sum().item<float>() << std::endl;

    // copy pointers correctly
    point_list = binningState.point_list;
    ranges = volState.ranges;
}

void RASTERIZER_PER_TILE::rasterize_gaussians_cuda_complex(
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

    //std::cout << "start rasterizing" << std::endl;
    results.fill_(0.0);

    const int X = mesh_resolutions[0].item<int>();
    const int Y = mesh_resolutions[1].item<int>();
    const int Z = mesh_resolutions[2].item<int>();
	dim3 tile_grid((X + BLOCK_X - 1) / BLOCK_X, (Y + BLOCK_Y - 1) / BLOCK_Y, (Z + BLOCK_Z - 1) / BLOCK_Z);
	dim3 pixel_block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    //std::cout << "geom buffer" << std::endl;

    torch::Tensor geometry_buffer = torch::empty({0}, scales.options());
    torch::Tensor binning_buffer = torch::empty({0}, scales.options());
    torch::Tensor volstate_buffer = torch::empty({0}, scales.options());

    uint32_t* point_list;
    uint2* ranges;
    generate_sorted_lists(
        geometry_buffer, binning_buffer, volstate_buffer,
        means, scales, mesh_lb, mesh_ub, mesh_resolutions, voxel_offset_factors, scale_multiplier, nr_gaussians, tile_grid, results,
        point_list, ranges
    );

    RASTERIZER_PER_TILE::FORWARD::rasterize_basic_complex(
        tile_grid, pixel_block,
        ranges,
        point_list,
        (glm::vec3*)means.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (glm::vec3*)scales.contiguous().data_ptr<float>(),
        (glm::vec4*)rotations.contiguous().data_ptr<float>(),
        (glm::vec3*)phases.contiguous().data_ptr<float>(),
        phases_add.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_lb.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_ub.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_resolutions.contiguous().data_ptr<float>(),
        (glm::vec3*)voxel_offset_factors.contiguous().data_ptr<float>(),
		use_phase_add_as_imag,
        scale_multiplier,
        (glm::vec2*)results.contiguous().data_ptr<float>()
    );

    //std::cout << "done" << std::endl;

    cudaDeviceSynchronize();
}

void RASTERIZER_PER_TILE::rasterize_gaussians_cuda_complex_backward(
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
    ) {

    grad_means.fill_(0.0);
    grad_opacities.fill_(0.0);
    grad_scales.fill_(0.0);
    grad_rotations.fill_(0.0);
    grad_phases.fill_(0.0);
    grad_phases_add.fill_(0.0);

    //std::cout << "start rasterizing" << std::endl;

    const int X = mesh_resolutions[0].item<int>();
    const int Y = mesh_resolutions[1].item<int>();
    const int Z = mesh_resolutions[2].item<int>();
	dim3 tile_grid((X + BLOCK_X - 1) / BLOCK_X, (Y + BLOCK_Y - 1) / BLOCK_Y, (Z + BLOCK_Z - 1) / BLOCK_Z);
	dim3 pixel_block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    //std::cout << "geom buffer" << std::endl;

    torch::Tensor geometry_buffer = torch::empty({0}, scales.options());
    torch::Tensor binning_buffer = torch::empty({0}, scales.options());
    torch::Tensor volstate_buffer = torch::empty({0}, scales.options());

    uint32_t* point_list;
    uint2* ranges;
    generate_sorted_lists(
        geometry_buffer, binning_buffer, volstate_buffer,
        means, scales, mesh_lb, mesh_ub, mesh_resolutions, voxel_offset_factors, scale_multiplier, nr_gaussians, tile_grid, grad_means,
        point_list, ranges
    );

    RASTERIZER_PER_TILE::BACKWARD::rasterize_basic_complex_backward(
        tile_grid, pixel_block,
        ranges,
        point_list,
        (glm::vec3*)means.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        (glm::vec3*)scales.contiguous().data_ptr<float>(),
        (glm::vec4*)rotations.contiguous().data_ptr<float>(),
        (glm::vec3*)phases.contiguous().data_ptr<float>(),
        phases_add.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_lb.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_ub.contiguous().data_ptr<float>(),
        (glm::vec3*)mesh_resolutions.contiguous().data_ptr<float>(),
        (glm::vec3*)voxel_offset_factors.contiguous().data_ptr<float>(),
		use_phase_add_as_imag, scale_multiplier, grad_padding_factor, grad_padding_const,
        (glm::vec2*)grad_results.contiguous().data_ptr<float>(),
        (glm::vec3*)grad_means.contiguous().data_ptr<float>(),
        grad_opacities.contiguous().data_ptr<float>(),
        (glm::vec3*)grad_scales.contiguous().data_ptr<float>(),
        (glm::vec4*)grad_rotations.contiguous().data_ptr<float>(),
        (glm::vec3*)grad_phases.contiguous().data_ptr<float>(),
        grad_phases_add.contiguous().data_ptr<float>()
    );

    //std::cout << "done" << std::endl;

    cudaDeviceSynchronize();

}

