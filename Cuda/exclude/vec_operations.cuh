//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC_OPERATIONS_CUH
#define ENGINE24_VEC_OPERATIONS_CUH

#include "mat_vec.cuh"

namespace Bcg::cuda {
    __host__ __device__ inline glm::mat2 outer(const glm::vec2 &a, const glm::vec2 &b) {
        return {a.x * b, a.y * b};
    }

    __host__ __device__ inline glm::mat3 outer(const glm::vec3 &a, const glm::vec3 &b) {
        return {a.x * b, a.y * b, a.z * b};
    }

    __host__ __device__ inline glm::mat4 outer(const glm::vec4 &a, const glm::vec4 &b) {
        return {a.x * b, a.y * b, a.z * b, a.w * b};
    }

    __host__ __device__ inline glm::mat2 as_diag(const glm::vec2 &v) {
        return {glm::vec2(v.x, 0), glm::vec2(0, v.y)};
    }

    __host__ __device__ inline glm::mat3 as_diag(const glm::vec3 &v) {
        return {glm::vec3(v.x, 0, 0), glm::vec3(0, v.y, 0), glm::vec3(0, 0, v.z)};
    }

    __host__ __device__ inline glm::mat4 as_diag(const glm::vec4 &v) {
        return {glm::vec4(v.x, 0, 0, 0), glm::vec4(0, v.y, 0, 0), glm::vec4(0, 0, v.z, 0), glm::vec4(0, 0, 0, v.w)};
    }
}
#endif //ENGINE24_VEC_OPERATIONS_CUH
