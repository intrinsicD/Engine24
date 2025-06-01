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
        vec3 max = vec3(-std::numeric_limits<float>::max());
        vec3 min = vec3(std::numeric_limits<float>::max());

        __device__ __host__
        bool empty() const noexcept {
            return max[0] < min[0] || max[1] < min[1] || max[2] < min[2];
        }
    };

} // lbvh
#endif// LBVH_AABB_CUH
