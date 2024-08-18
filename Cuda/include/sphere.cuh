//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_SPHERE_CUH
#define ENGINE24_SPHERE_CUH

#include "vec4.cuh"
#include "aabb.cuh"
#include <thrust/swap.h>
#include <cmath>

namespace Bcg::cuda {
    struct sphere {
        vec4 center_radius;

        sphere() noexcept = default;

        __device__ __host__
        sphere(const vec3 &center, float radius) noexcept: center_radius(center.x, center.y, center.z, radius) {}
    };

    __device__ __host__
    inline bool intersects(const sphere &sphere, const aabb &box) noexcept {
        float sqDist = 0.0f;

        if (sphere.center_radius.x < box.lower.x)
            sqDist += (box.lower.x - sphere.center_radius.x) *
                      (box.lower.x - sphere.center_radius.x);
        if (sphere.center_radius.x > box.upper.x)
            sqDist += (sphere.center_radius.x - box.upper.x) *
                      (sphere.center_radius.x - box.upper.x);

        if (sphere.center_radius.y < box.lower.y)
            sqDist += (box.lower.y - sphere.center_radius.y) *
                      (box.lower.y - sphere.center_radius.y);
        if (sphere.center_radius.y > box.upper.y)
            sqDist += (sphere.center_radius.y - box.upper.y) *
                      (sphere.center_radius.y - box.upper.y);

        if (sphere.center_radius.z < box.lower.z)
            sqDist += (box.lower.z - sphere.center_radius.z) *
                      (box.lower.z - sphere.center_radius.z);
        if (sphere.center_radius.z > box.upper.z)
            sqDist += (sphere.center_radius.z - box.upper.z) *
                      (sphere.center_radius.z - box.upper.z);

        return sqDist <= sphere.center_radius.w * sphere.center_radius.w;
    }

    __device__ __host__
    inline bool intersects(const aabb &lhs, const sphere &rhs) noexcept {
        return intersects(rhs, lhs);
    }

    __device__ __host__
    inline float mindist(const sphere &sphere, const vec3 &rhs) noexcept {
        const float sqDist =
                (rhs.x - sphere.center_radius.x) * (rhs.x - sphere.center_radius.x) +
                (rhs.y - sphere.center_radius.y) * (rhs.y - sphere.center_radius.y) +
                (rhs.z - sphere.center_radius.z) * (rhs.z - sphere.center_radius.z);
        return ::fabs(::sqrtf(sqDist) - sphere.center_radius.w);
    }

    __device__ __host__
    inline float minmaxdist(const sphere &sphere, const vec3 &point) noexcept {
        return mindist(sphere, point) + sphere.center_radius.w;
    }

    struct sphere_getter {
        __device__ __host__
        sphere operator()(const vec3 center, float radius) const noexcept {
            sphere retval;
            retval.center_radius.x = center.x;
            retval.center_radius.y = center.y;
            retval.center_radius.z = center.z;
            retval.center_radius.w = radius;
            return retval;
        }
    };
}

#endif //ENGINE24_SPHERE_CUH
