//
// Created by alex on 6/1/25.
//

#ifndef SPHERE_UTILS_CUH
#define SPHERE_UTILS_CUH

#include "sphere.cuh"
#include "aabb_utils.cuh"

namespace Bcg::cuda {
    struct sphere_getter {
        __device__ __host__
        sphere operator()(const vec3 center, float radius) const noexcept {
            return {center, radius};
        }

        __device__ __host__
        sphere operator()(const vec4 &v) const noexcept {
            return {{v[0], v[1], v[2]}, v[3]};
        }

        __device__ __host__
        sphere operator()(const aabb &aabb) const noexcept {
            return {centroid(aabb), 0.5f * length(aabb.max - aabb.min)};
        }
    };

    __device__ __host__
    inline bool intersects(const sphere &s, const aabb &b) noexcept {
        // find squared distance from sphere center to AABB
        float dist2 = 0.0f;

        // X axis
        if (s.center[0] < b.min[0]) {
            float d = b.min[0] - s.center[0];
            dist2 += d*d;
        } else if (s.center[0] > b.max[0]) {
            float d = s.center[0] - b.max[0];
            dist2 += d*d;
        }

        // Y axis
        if (s.center[1] < b.min[1]) {
            float d = b.min[1] - s.center[1];
            dist2 += d*d;
        } else if (s.center[1] > b.max[1]) {
            float d = s.center[1] - b.max[1];
            dist2 += d*d;
        }

        // Z axis
        if (s.center[2] < b.min[2]) {
            float d = b.min[2] - s.center[2];
            dist2 += d*d;
        } else if (s.center[2] > b.max[2]) {
            float d = s.center[2] - b.max[2];
            dist2 += d*d;
        }

        return dist2 <= (s.radius * s.radius);
    }

    __device__ __host__
    inline bool intersects(const aabb &lhs, const sphere &rhs) noexcept {
        return intersects(rhs, lhs);
    }

    __device__ __host__
    inline float mindist(const sphere &sphere, const vec3 &rhs) noexcept {
        const float sqDist =
                (rhs[0] - sphere.center[0]) * (rhs[0] - sphere.center[0]) +
                (rhs[1] - sphere.center[1]) * (rhs[1] - sphere.center[1]) +
                (rhs[2] - sphere.center[2]) * (rhs[2] - sphere.center[2]);
        return ::fabs(::sqrtf(sqDist) - sphere.radius);
    }

    __device__ __host__
    inline float minmaxdist(const sphere &sphere, const vec3 &point) noexcept {
        return mindist(sphere, point) + sphere.radius;
    }


}

#endif //SPHERE_UTILS_CUH
