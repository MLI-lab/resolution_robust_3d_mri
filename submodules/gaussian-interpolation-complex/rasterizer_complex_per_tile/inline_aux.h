
#pragma once

#include "config.h"
#include <glm/glm.hpp>

__forceinline__ __device__ void getPixelBounds(const glm::vec3& mesh_resolutions, float max_scale, const glm::vec3& ref_point, const glm::vec3& stepsize, glm::vec3& l, glm::vec3& u) {
	const int stepsize_x_voxel = glm::ceil(max_scale / stepsize.x);
	const int stepsize_y_voxel = glm::ceil(max_scale / stepsize.y);
	const int stepsize_z_voxel = glm::ceil(max_scale / stepsize.z);

    // pixel boundaries
    // ref point might be negative, so l and u are capped to be within the pixel grid
    //l = {
        //glm::min(glm::max((int)ref_point.x - stepsize_x_voxel, 0), (int)mesh_resolutions.x-1),
        //glm::min(glm::max((int)ref_point.y - stepsize_y_voxel, 0), (int)mesh_resolutions.y-1),
        //glm::min(glm::max((int)ref_point.z - stepsize_z_voxel, 0), (int)mesh_resolutions.z-1)
    //};

    //u = {
        //glm::max(glm::min((int)ref_point.x + stepsize_x_voxel, (int)mesh_resolutions.x-1), 0),
        //glm::max(glm::min((int)ref_point.y + stepsize_y_voxel, (int)mesh_resolutions.y-1), 0),
        //glm::max(glm::min((int)ref_point.z + stepsize_z_voxel, (int)mesh_resolutions.z-1), 0)
    //};

    l = {
        ref_point.x - stepsize_x_voxel,
        ref_point.y - stepsize_y_voxel,
        ref_point.z - stepsize_z_voxel
    };

    u = {
        ref_point.x + stepsize_x_voxel,
        ref_point.y + stepsize_y_voxel,
        ref_point.z + stepsize_z_voxel
    };

    // l and u might now also be negative (if out of view)
}

// 0 1 2 3 4 5 6 7 | 8 9 10
// suppose I have a gaussian with ref point 7 and stepsize voxel 1
// then l = ref_point - 1 = 6 => rect_min = 6 / 8 = 0
// u = rep_point + 1 = 8 => rect_max = (8 + 7) / 8 = 1
// -> this case would be ignored for replication??

// if stepsize voxel is 2
// then l = ref_point - 2 = 5 => rect_min = 5 / 8 = 0
// u = ref_point + 2 = 9 => rect_max = (9 + 7) / 8 = 2

__forceinline__ __device__ void getTileBounds(const glm::vec3& pl, const glm::vec3& pu, dim3 grid, glm::vec3& rect_min, glm::vec3& rect_max)
{
	//rect_min = {
		//min(grid.x, max((int)0, ( ((int)pl.x) / BLOCK_X))),
		//min(grid.y, max((int)0, ( ((int)pl.y) / BLOCK_Y))),
		//min(grid.z, max((int)0, ( ((int)pl.z) / BLOCK_Z)))
	//};
	//rect_max = {
		//min(grid.x, max((int)0, ( ((int)pu.x) + BLOCK_X - 1) / BLOCK_X) ),
		//min(grid.y, max((int)0, ( ((int)pu.y) + BLOCK_Y - 1) / BLOCK_Y) ),
		//min(grid.z, max((int)0, ( ((int)pu.z) + BLOCK_Z - 1) / BLOCK_Z) )
	//};

    // l and u could be negative (if out of view)
    // rect_max is now inclusive
    rect_min = {
        glm::floor(pl.x / BLOCK_X),
        glm::floor(pl.y / BLOCK_Y),
        glm::floor(pl.z / BLOCK_Z)
    };
	rect_max = {
        glm::floor(pu.x / BLOCK_X),
        glm::floor(pu.y / BLOCK_Y),
        glm::floor(pu.z / BLOCK_Z)
	};

    // tile min and max are now inclusive and might also be negative

    // rect_min == recht_max can often happen, but is not a problem.
    // 
}

__forceinline__ __device__ void getStepsizeCoord(const glm::vec3& mesh_resolutions, const glm::vec3& mesh_lb, const glm::vec3& mesh_ub, glm::vec3& stepsize) {
    stepsize = {
	    (mesh_ub.x - mesh_lb.x) / ((int)mesh_resolutions.x - 1),
	    (mesh_ub.y - mesh_lb.y) / ((int)mesh_resolutions.y - 1),
	    (mesh_ub.z - mesh_lb.z) / ((int)mesh_resolutions.z - 1)
    };
}

__forceinline__ __device__ void getGridRefPointPixel(const glm::vec3 mean, const glm::vec3& mesh_lb, const glm::vec3& stepsize, glm::vec3& ref_point) {
    // result might be negative if gaussian mean is not within the mesh bounds
    ref_point = {
	    (int)glm::floor( (mean.x - mesh_lb.x) / stepsize.x ),
	    (int)glm::floor( (mean.y - mesh_lb.y) / stepsize.y ),
	    (int)glm::floor( (mean.z - mesh_lb.z) / stepsize.z )
    };
}

__forceinline__ __device__ void getTileBounds(
	const glm::vec3& mesh_resolutions, const glm::vec3& mesh_lb, const glm::vec3& mesh_ub, const glm::vec3& mean, const glm::vec3& scale, float scale_modifier, dim3 tile_grid,
	glm::vec3& rect_min, glm::vec3& rect_max) {
	
	const float max_scale = scale_modifier * glm::max(scale.x, glm::max(scale.y, scale.z));

    glm::vec3 stepsize;
    getStepsizeCoord(mesh_resolutions, mesh_lb, mesh_ub, stepsize);

    glm::vec3 ref_point;
    getGridRefPointPixel(mean, mesh_lb, stepsize, ref_point);

    glm::vec3 l, u;
    getPixelBounds(mesh_resolutions, max_scale, ref_point, stepsize, l, u);

    getTileBounds(l, u, tile_grid, rect_min, rect_max);
}