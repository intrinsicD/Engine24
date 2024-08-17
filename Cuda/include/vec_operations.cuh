//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC_OPERATIONS_CUH
#define ENGINE24_VEC_OPERATIONS_CUH

#include "vec4.cuh"

#include "mat4.cuh"

namespace Bcg::cuda {
    __host__ __device__ inline mat2 outer(const vec2 &a, const vec2 &b) {
        return {a.x * b, a.y * b};
    }

    __host__ __device__ inline mat3 outer(const vec3 &a, const vec3 &b) {
        return {a.x * b, a.y * b, a.z * b};
    }

    __host__ __device__ inline mat4 outer(const vec4 &a, const vec4 &b) {
        return {a.x * b, a.y * b, a.z * b, a.w * b};
    }

    __host__ __device__ inline mat2 as_diag(const vec2 &v) {
        return {vec2(v.x, 0), vec2(0, v.y)};
    }

    __host__ __device__ inline mat3 as_diag(const vec3 &v) {
        return {vec3(v.x, 0, 0), vec3(0, v.y, 0), vec3(0, 0, v.z)};
    }

    __host__ __device__ inline mat4 as_diag(const vec4 &v) {
        return {vec4(v.x, 0, 0, 0), vec4(0, v.y, 0, 0), vec4(0, 0, v.z, 0), vec4(0, 0, 0, v.w)};
    }
}
#endif //ENGINE24_VEC_OPERATIONS_CUH
