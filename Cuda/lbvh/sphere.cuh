//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_SPHERE_CUH
#define ENGINE24_SPHERE_CUH

#include "utility.cuh"
#include "aabb.cuh"
#include <thrust/swap.h>
#include <cmath>

namespace lbvh {
    template<typename T>
    struct sphere {
        typename vector_of<T>::type center;
        T radius;
    };

    template<typename T>
    __device__ __host__
    inline bool intersects(const sphere<T> &sphere, const aabb<T> &box) noexcept {
        float sqDist = 0.0f;

        if (sphere.center.x < box.lower.x) sqDist += (box.lower.x - sphere.center.x) * (box.lower.x - sphere.center.x);
        if (sphere.center.x > box.upper.x) sqDist += (sphere.center.x - box.upper.x) * (sphere.center.x - box.upper.x);

        if (sphere.center.y < box.lower.y) sqDist += (box.lower.y - sphere.center.y) * (box.lower.y - sphere.center.y);
        if (sphere.center.y > box.upper.y) sqDist += (sphere.center.y - box.upper.y) * (sphere.center.y - box.upper.y);

        if (sphere.center.z < box.lower.z) sqDist += (box.lower.z - sphere.center.z) * (box.lower.z - sphere.center.z);
        if (sphere.center.z > box.upper.z) sqDist += (sphere.center.z - box.upper.z) * (sphere.center.z - box.upper.z);

        return sqDist <= sphere.radius * sphere.radius;
    }

    template<typename T>
    __device__ __host__
    inline bool intersects(const aabb<T> &lhs, const sphere <T> &rhs) noexcept {
        return intersect(rhs, lhs);
    }

    __device__ __host__
    inline float mindist(const sphere<float> &sphere, const float4 &rhs) noexcept {
        const float sqDist =
                (rhs.x - sphere.center.x) * (rhs.x - sphere.center.x) +
                (rhs.y - sphere.center.y) * (rhs.y - sphere.center.y) +
                (rhs.z - sphere.center.z) * (rhs.z - sphere.center.z);
        return ::fabs(::sqrtf(sqDist) - sphere.radius);
    }

    __device__ __host__
    inline double mindist(const sphere<double> &sphere, const double4 &rhs) noexcept {
        const double sqDist =
                (rhs.x - sphere.center.x) * (rhs.x - sphere.center.x) +
                (rhs.y - sphere.center.y) * (rhs.y - sphere.center.y) +
                (rhs.z - sphere.center.z) * (rhs.z - sphere.center.z);
        return ::abs(::sqrt(sqDist) - sphere.radius);
    }

    __device__ __host__
    inline float minmaxdist(const sphere<float> &sphere, const float4 &point) noexcept {
        return mindist(sphere, point) + sphere.radius;
    }

    __device__ __host__
    inline double minmaxdist(const sphere<double> &sphere, const double4 &point) noexcept {
        return mindist(sphere, point) + sphere.radius;
    }

    struct sphere_getter {
        __device__ __host__
        lbvh::sphere<float> operator()(const float4 center, float radius) const noexcept {
            lbvh::sphere<float> retval;
            retval.center = center;
            retval.radius = radius;
            return retval;
        }
    };
}

#endif //ENGINE24_SPHERE_CUH
