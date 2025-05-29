#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH

#include "mat_vec.cuh"
#include "utility.cuh"
#include <thrust/swap.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <cmath>

namespace Bcg::cuda {
    struct aabb {
        vec3 max;
        vec3 min;
    };

    template<typename Object>
    struct aabb_getter;

    template<>
    struct aabb_getter<vec3> {
        __device__ __host__
        aabb operator()(const vec3 &v) const noexcept {
            return {v, v};
        }
    };

    __device__ __host__
    inline bool intersects(const aabb &lhs, const aabb &rhs) noexcept {
        // Check if the two AABBs overlap
        if (lhs.max[0] < rhs.min[0] || rhs.max[0] < lhs.min[0]) { return false; }
        if (lhs.max[1] < rhs.min[1] || rhs.max[1] < lhs.min[1]) { return false; }
        if (lhs.max[2] < rhs.min[2] || rhs.max[2] < lhs.min[2]) { return false; }
        return true;
    }

    __device__ __host__
    inline aabb merge(const aabb &lhs, const aabb &rhs) noexcept {
        aabb merged;
        merged.max[0] = ::fmaxf(lhs.max[0], rhs.max[0]);
        merged.max[1] = ::fmaxf(lhs.max[1], rhs.max[1]);
        merged.max[2] = ::fmaxf(lhs.max[2], rhs.max[2]);
        merged.min[0] = ::fminf(lhs.min[0], rhs.min[0]);
        merged.min[1] = ::fminf(lhs.min[1], rhs.min[1]);
        merged.min[2] = ::fminf(lhs.min[2], rhs.min[2]);
        return merged;
    }

    struct aabb_merger{
        __device__ __host__
        aabb operator()(const aabb &lhs, const aabb &rhs) const noexcept {
            return merge(lhs, rhs);
        }
    };

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent

    __device__ __host__
    inline float mindist(const aabb &lhs, const vec3 &rhs) noexcept {
        const float dx = ::fminf(lhs.max[0], ::fmaxf(lhs.min[0], rhs[0])) - rhs[0];
        const float dy = ::fminf(lhs.max[1], ::fmaxf(lhs.min[1], rhs[1])) - rhs[1];
        const float dz = ::fminf(lhs.max[2], ::fmaxf(lhs.min[2], rhs[2])) - rhs[2];
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __host__
    inline float minmaxdist(const aabb &lhs, const vec3 &rhs) noexcept {
        vec3 lower_diff = lhs.min - rhs;
        vec3 upper_diff = lhs.max - rhs;
        vec3 rm_sq = vec3(lower_diff[0] * lower_diff[0], lower_diff[1] * lower_diff[1], lower_diff[2] * lower_diff[2]);
        vec3 rM_sq = vec3(upper_diff[0] * upper_diff[0], upper_diff[1] * upper_diff[1], upper_diff[2] * upper_diff[2]);

        if ((lhs.max[0] + lhs.min[0]) * 0.5f < rhs[0]) {
            thrust::swap(rm_sq[0], rM_sq[0]);
        }
        if ((lhs.max[1] + lhs.min[1]) * 0.5f < rhs[1]) {
            thrust::swap(rm_sq[1], rM_sq[1]);
        }
        if ((lhs.max[2] + lhs.min[2]) * 0.5f < rhs[2]) {
            thrust::swap(rm_sq[2], rM_sq[2]);
        }

        const float dx = rm_sq[0] + rM_sq[1] + rM_sq[2];
        const float dy = rM_sq[0] + rm_sq[1] + rM_sq[2];
        const float dz = rM_sq[0] + rM_sq[1] + rm_sq[2];
        return ::fminf(dx, ::fminf(dy, dz));
    }

    __device__ __host__
    inline vec3 centroid(const aabb &box) noexcept {
        return (box.max + box.min) * 0.5f;
    }
} // lbvh
#endif// LBVH_AABB_CUH
