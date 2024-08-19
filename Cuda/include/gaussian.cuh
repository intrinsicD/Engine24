//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_GAUSSIAN_CUH
#define ENGINE24_GAUSSIAN_CUH

#include "vec3.cuh"
#include "mat3.cuh"
#include "math.cuh"

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

    __device__ __host__ inline float
    kullback_leibler_divergence(const vec3 &mean_p, const mat3 &cov_p,
                                const vec3 &mean_q, const mat3 &cov_q, const mat3 &inv_cov_q) {
        double trace_term = (inv_cov_q * cov_p).trace();
        vec3 diff = mean_q - mean_p;
        double quadratic_term = diff.dot(inv_cov_q * diff);
        double log_det_term = logf(cov_q.determinant() / cov_p.determinant());
        double kl_divergence = 0.5f * (trace_term + quadratic_term - 3 + log_det_term);
        return kl_divergence;
    }

    __device__ __host__ inline mat3
    conditionCov(const mat3 &cov, vec3 &outEvalues, float epsilon = 1e-10f) {
        mat3 newCov;
        float abseps = fabsf(epsilon);

        // condition diagonal elements
        newCov.col0.x = fmaxf(cov.col0.x, abseps);
        newCov.col1.y = fmaxf(cov.col1.y, abseps);
        newCov.col2.z = fmaxf(cov.col2.z, abseps);

        // condition off diagonal elements
        float sx = sqrtf(cov.col0.x);
        float sy = sqrtf(cov.col1.y);
        float sz = sqrtf(cov.col2.z);

        bool conditioned = false;
        for (float rho = 0.99f; rho >= 0; rho -= 0.01f) {
            float rxy = rho * sx * sy;
            float rxz = rho * sx * sz;
            float ryz = rho * sy * sz;
            newCov.col1.x = clamp(cov.col1.x, -rxy, rxy);
            newCov.col2.x = clamp(cov.col2.x, -rxz, rxz);
            newCov.col2.y = clamp(cov.col2.y, -ryz, ryz);

            newCov.col0.y = newCov.col1.x;
            newCov.col0.z = newCov.col2.x;
            newCov.col1.z = newCov.col2.y;

            // Check
            mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);
            if (outEvalues.x > 0.0f && outEvalues.y > 0.0f && outEvalues.z > 0.0f) {
                conditioned = true;
                break;
            }
        }

        // Check
        if (!conditioned) {
            // Add a small multiple of the identity matrix as a fallback
            newCov = newCov + mat3::identity() * abseps;

            // Recompute eigenvalues for the final conditioned matrix
            mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);

            // Check and warn
            if (outEvalues.x <= 0.0f || outEvalues.y <= 0.0f || outEvalues.z <= 0.0f) {
                printf("Warning: cov still non-psd despite conditioning! det: %f\n", cov.determinant());
                printf("evals: %f, %f, %f\n", outEvalues.x, outEvalues.y, outEvalues.z);
            }
        }
        return newCov;
    }

    __device__ __host__ inline mat3 conditionCov(const mat3 &cov, float epsilon = 1e-10f) {
        vec3 evd;
        return conditionCov(cov, evd, epsilon);
    }

}

#endif //ENGINE24_GAUSSIAN_CUH
