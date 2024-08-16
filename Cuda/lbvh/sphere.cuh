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
        typename vector_of<T>::type center_radius;
    };

    template<typename T>
    __device__ __host__
    inline bool intersects(const sphere<T> &sphere, const aabb<T> &box) noexcept {
        float sqDist = 0.0f;

        if (sphere.center_radius.x < box.lower.x) sqDist += (box.lower.x - sphere.center_radius.x) *
                                                            (box.lower.x - sphere.center_radius.x);
        if (sphere.center_radius.x > box.upper.x) sqDist += (sphere.center_radius.x - box.upper.x) *
                                                            (sphere.center_radius.x - box.upper.x);

        if (sphere.center_radius.y < box.lower.y) sqDist += (box.lower.y - sphere.center_radius.y) *
                                                            (box.lower.y - sphere.center_radius.y);
        if (sphere.center_radius.y > box.upper.y) sqDist += (sphere.center_radius.y - box.upper.y) *
                                                            (sphere.center_radius.y - box.upper.y);

        if (sphere.center_radius.z < box.lower.z) sqDist += (box.lower.z - sphere.center_radius.z) *
                                                            (box.lower.z - sphere.center_radius.z);
        if (sphere.center_radius.z > box.upper.z) sqDist += (sphere.center_radius.z - box.upper.z) *
                                                            (sphere.center_radius.z - box.upper.z);

        return sqDist <= sphere.center_radius.w * sphere.center_radius.w;
    }

    template<typename T>
    __device__ __host__
    inline bool intersects(const aabb<T> &lhs, const sphere <T> &rhs) noexcept {
        return intersect(rhs, lhs);
    }

    __device__ __host__
    inline float mindist(const sphere<float> &sphere, const float4 &rhs) noexcept {
        const float sqDist =
                (rhs.x - sphere.center_radius.x) * (rhs.x - sphere.center_radius.x) +
                (rhs.y - sphere.center_radius.y) * (rhs.y - sphere.center_radius.y) +
                (rhs.z - sphere.center_radius.z) * (rhs.z - sphere.center_radius.z);
        return ::fabs(::sqrtf(sqDist) - sphere.center_radius.w);
    }

    __device__ __host__
    inline double mindist(const sphere<double> &sphere, const double4 &rhs) noexcept {
        const double sqDist =
                (rhs.x - sphere.center_radius.x) * (rhs.x - sphere.center_radius.x) +
                (rhs.y - sphere.center_radius.y) * (rhs.y - sphere.center_radius.y) +
                (rhs.z - sphere.center_radius.z) * (rhs.z - sphere.center_radius.z);
        return ::abs(::sqrt(sqDist) - sphere.center_radius.w);
    }

    __device__ __host__
    inline float minmaxdist(const sphere<float> &sphere, const float4 &point) noexcept {
        return mindist(sphere, point) + sphere.center_radius.w;
    }

    __device__ __host__
    inline double minmaxdist(const sphere<double> &sphere, const double4 &point) noexcept {
        return mindist(sphere, point) + sphere.center_radius.w;
    }

    struct sphere_getter {
        __device__ __host__
        lbvh::sphere<float> operator()(const float4 center, float radius) const noexcept {
            lbvh::sphere<float> retval;
            retval.center_radius.x = center.x;
            retval.center_radius.y = center.y;
            retval.center_radius.z = center.z;
            retval.center_radius.w = radius;
            return retval;
        }
    };
}

#endif //ENGINE24_SPHERE_CUH
