#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH

#include "utility.cuh"
#include "vec3.cuh"
#include "functors.cuh"
#include <thrust/swap.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <cmath>

namespace Bcg::cuda {
    struct aabb {
        vec3 upper;
        vec3 lower;
    };

    struct aabb_getter {
        __device__ __host__
        aabb operator()(const vec3 v) const noexcept {
            return {v, v};
        }
    };

    __device__ __host__
    inline bool intersects(const aabb &lhs, const aabb &rhs) noexcept {
        if (lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x) { return false; }
        if (lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y) { return false; }
        if (lhs.upper.z < rhs.lower.z || rhs.upper.z < lhs.lower.z) { return false; }
        return true;
    }

    __device__ __host__
    inline aabb merge(const aabb &lhs, const aabb &rhs) noexcept {
        aabb merged;
        merged.upper.x = ::fmaxf(lhs.upper.x, rhs.upper.x);
        merged.upper.y = ::fmaxf(lhs.upper.y, rhs.upper.y);
        merged.upper.z = ::fmaxf(lhs.upper.z, rhs.upper.z);
        merged.lower.x = ::fminf(lhs.lower.x, rhs.lower.x);
        merged.lower.y = ::fminf(lhs.lower.y, rhs.lower.y);
        merged.lower.z = ::fminf(lhs.lower.z, rhs.lower.z);
        return merged;
    }

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent

    __device__ __host__
    inline float mindist(const aabb &lhs, const vec3 &rhs) noexcept {
        const float dx = ::fminf(lhs.upper.x, ::fmaxf(lhs.lower.x, rhs.x)) - rhs.x;
        const float dy = ::fminf(lhs.upper.y, ::fmaxf(lhs.lower.y, rhs.y)) - rhs.y;
        const float dz = ::fminf(lhs.upper.z, ::fmaxf(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __host__
    inline float minmaxdist(const aabb &lhs, const vec3 &rhs) noexcept {
        vec3 lower_diff = lhs.lower - rhs;
        vec3 upper_diff = lhs.upper - rhs;
        vec3 rm_sq = vec3(lower_diff.x * lower_diff.x, lower_diff.y * lower_diff.y, lower_diff.z * lower_diff.z);
        vec3 rM_sq = vec3(upper_diff.x * upper_diff.x, upper_diff.y * upper_diff.y, upper_diff.z * upper_diff.z);

        if ((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x) {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if ((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y) {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        if ((lhs.upper.z + lhs.lower.z) * 0.5f < rhs.z) {
            thrust::swap(rm_sq.z, rM_sq.z);
        }

        const float dx = rm_sq.x + rM_sq.y + rM_sq.z;
        const float dy = rM_sq.x + rm_sq.y + rM_sq.z;
        const float dz = rM_sq.x + rM_sq.y + rm_sq.z;
        return ::fminf(dx, ::fminf(dy, dz));
    }

    __device__ __host__
    inline vec3 centroid(const aabb &box) noexcept {
        return (box.upper + box.lower) * 0.5;
    }
} // lbvh
#endif// LBVH_AABB_CUH
