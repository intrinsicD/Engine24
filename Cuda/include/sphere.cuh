//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_SPHERE_CUH
#define ENGINE24_SPHERE_CUH

#include "mat_vec.cuh"
#include "aabb.cuh"

namespace Bcg::cuda {
    struct sphere {
        vec3 center;
        float radius;

        sphere() noexcept = default;

        __device__ __host__
        sphere(const vec3 &center, float radius) noexcept
            : center(center), radius(radius) {}
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

    struct sphere_getter {
        __device__ __host__
        sphere operator()(const vec3 center, float radius) const noexcept {
            return {center, radius};
        }
    };
}

#endif //ENGINE24_SPHERE_CUH
