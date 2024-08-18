//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_GAUSSIAN_CUH
#define ENGINE24_GAUSSIAN_CUH

#include "vec3.cuh"
#include "mat3.cuh"

namespace Bcg::cuda {
    struct gaussian {
        vec3 mean;
        mat3 cov;
    };

    __device__ __host__ inline float gaussian_pdf(const vec3 &mean, const vec3 &x, float det, const mat3 &inv_cov) {
        vec3 diff = x - mean;
        float exponent = -0.5f * diff.dot(inv_cov * diff);
        return expf(exponent) / sqrtf(det * 8 * CUDART_PI);
    }

    __device__ __host__ float kullback_leibler_divergence(const vec3 &mean_p, const mat3 &cov_p, const mat3 &inv_cov_p,
                                                          const vec3 &mean_q, const mat3 &cov_q, const mat3 &inv_cov_q) {
        float trace = (inv_cov_q * cov_p).trace();
        vec3 diff = mean_q - mean_p;
        float exponent = 0.5f * (trace + diff.dot(inv_cov_q * diff) - 3);
        return 0.5f * exponent;
    }


}

#endif //ENGINE24_GAUSSIAN_CUH
