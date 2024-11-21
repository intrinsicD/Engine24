//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_SPHERE_CUH
#define ENGINE24_SPHERE_CUH

#include "glm/glm.hpp"
#include "aabb.cuh"
#include <thrust/swap.h>
#include <cmath>

namespace Bcg::cuda {
    struct sphere {
        glm::vec4 center_radius;

        sphere() noexcept = default;

        __device__ __host__
        sphere(const glm::vec3 &center, float radius) noexcept: center_radius(center.x, center.y, center.z, radius) {}
    };

    __device__ __host__
    inline bool intersects(const sphere &sphere, const aabb &box) noexcept {
        float sqDist = 0.0f;

        if (sphere.center_radius.x < box.min.x)
            sqDist += (box.min.x - sphere.center_radius.x) *
                      (box.min.x - sphere.center_radius.x);
        if (sphere.center_radius.x > box.max.x)
            sqDist += (sphere.center_radius.x - box.max.x) *
                      (sphere.center_radius.x - box.max.x);

        if (sphere.center_radius.y < box.min.y)
            sqDist += (box.min.y - sphere.center_radius.y) *
                      (box.min.y - sphere.center_radius.y);
        if (sphere.center_radius.y > box.max.y)
            sqDist += (sphere.center_radius.y - box.max.y) *
                      (sphere.center_radius.y - box.max.y);

        if (sphere.center_radius.z < box.min.z)
            sqDist += (box.min.z - sphere.center_radius.z) *
                      (box.min.z - sphere.center_radius.z);
        if (sphere.center_radius.z > box.max.z)
            sqDist += (sphere.center_radius.z - box.max.z) *
                      (sphere.center_radius.z - box.max.z);

        return sqDist <= sphere.center_radius.w * sphere.center_radius.w;
    }

    __device__ __host__
    inline bool intersects(const aabb &lhs, const sphere &rhs) noexcept {
        return intersects(rhs, lhs);
    }

    __device__ __host__
    inline float mindist(const sphere &sphere, const glm::vec3 &rhs) noexcept {
        const float sqDist =
                (rhs.x - sphere.center_radius.x) * (rhs.x - sphere.center_radius.x) +
                (rhs.y - sphere.center_radius.y) * (rhs.y - sphere.center_radius.y) +
                (rhs.z - sphere.center_radius.z) * (rhs.z - sphere.center_radius.z);
        return ::fabs(::sqrtf(sqDist) - sphere.center_radius.w);
    }

    __device__ __host__
    inline float minmaxdist(const sphere &sphere, const glm::vec3 &point) noexcept {
        return mindist(sphere, point) + sphere.center_radius.w;
    }

    struct sphere_getter {
        __device__ __host__
        sphere operator()(const glm::vec3 center, float radius) const noexcept {
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
