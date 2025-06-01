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

}

#endif //ENGINE24_SPHERE_CUH
