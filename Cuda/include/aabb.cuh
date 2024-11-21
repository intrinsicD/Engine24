#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH

#include "utility.cuh"
#include <thrust/swap.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <cmath>
#include "glm/glm.hpp"

namespace Bcg::cuda {
    struct aabb {
        glm::vec3 max;
        glm::vec3 min;
    };

    template<typename Object>
    struct aabb_getter;

    template<>
    struct aabb_getter<glm::vec3> {
        __device__ __host__
        aabb operator()(const glm::vec3 v) const noexcept {
            return {v, v};
        }
    };

    __device__ __host__
    inline bool intersects(const aabb &lhs, const aabb &rhs) noexcept {
        if (lhs.max.x < rhs.min.x || rhs.max.x < lhs.min.x) { return false; }
        if (lhs.max.y < rhs.min.y || rhs.max.y < lhs.min.y) { return false; }
        if (lhs.max.z < rhs.min.z || rhs.max.z < lhs.min.z) { return false; }
        return true;
    }

    __device__ __host__
    inline aabb merge(const aabb &lhs, const aabb &rhs) noexcept {
        aabb merged;
        merged.max.x = ::fmaxf(lhs.max.x, rhs.max.x);
        merged.max.y = ::fmaxf(lhs.max.y, rhs.max.y);
        merged.max.z = ::fmaxf(lhs.max.z, rhs.max.z);
        merged.min.x = ::fminf(lhs.min.x, rhs.min.x);
        merged.min.y = ::fminf(lhs.min.y, rhs.min.y);
        merged.min.z = ::fminf(lhs.min.z, rhs.min.z);
        return merged;
    }

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent

    __device__ __host__
    inline float mindist(const aabb &lhs, const glm::vec3 &rhs) noexcept {
        const float dx = ::fminf(lhs.max.x, ::fmaxf(lhs.min.x, rhs.x)) - rhs.x;
        const float dy = ::fminf(lhs.max.y, ::fmaxf(lhs.min.y, rhs.y)) - rhs.y;
        const float dz = ::fminf(lhs.max.z, ::fmaxf(lhs.min.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    }

    __device__ __host__
    inline float minmaxdist(const aabb &lhs, const glm::vec3 &rhs) noexcept {
        glm::vec3 lower_diff = lhs.min - rhs;
        glm::vec3 upper_diff = lhs.max - rhs;
        glm::vec3 rm_sq = glm::vec3(lower_diff.x * lower_diff.x, lower_diff.y * lower_diff.y, lower_diff.z * lower_diff.z);
        glm::vec3 rM_sq = glm::vec3(upper_diff.x * upper_diff.x, upper_diff.y * upper_diff.y, upper_diff.z * upper_diff.z);

        if ((lhs.max.x + lhs.min.x) * 0.5f < rhs.x) {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if ((lhs.max.y + lhs.min.y) * 0.5f < rhs.y) {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        if ((lhs.max.z + lhs.min.z) * 0.5f < rhs.z) {
            thrust::swap(rm_sq.z, rM_sq.z);
        }

        const float dx = rm_sq.x + rM_sq.y + rM_sq.z;
        const float dy = rM_sq.x + rm_sq.y + rM_sq.z;
        const float dz = rM_sq.x + rM_sq.y + rm_sq.z;
        return ::fminf(dx, ::fminf(dy, dz));
    }

    __device__ __host__
    inline glm::vec3 centroid(const aabb &box) noexcept {
        return (box.max + box.min) * 0.5f;
    }
} // lbvh
#endif// LBVH_AABB_CUH
