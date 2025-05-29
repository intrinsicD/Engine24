//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_GAUSSIAN_CUH
#define ENGINE24_GAUSSIAN_CUH

#include "mat_vec.cuh"
#include "MatUtils.h"
#include "math.cuh"

namespace Bcg::cuda {
    struct gaussian {
        vec3 mean;
        mat3 cov;
    };

    __device__ __host__ inline float gaussian_pdf(const vec3 &mean, const vec3 &x, float det, const mat3 &inv_cov) {
        vec3 diff = x - mean;
        float exponent = -0.5f * dot(diff, inv_cov * diff);
        return expf(exponent) / sqrtf(det * 8 * CUDART_PI);
    }

    __device__ __host__ inline float
    kullback_leibler_divergence(const vec3 &mean_p, const mat3 &cov_p,
                                const vec3 &mean_q, const mat3 &cov_q, const mat3 &inv_cov_q) {
        float trace_term = trace(inv_cov_q * cov_p);
        vec3 diff = mean_q - mean_p;
        float quadratic_term = dot(diff, inv_cov_q * diff);
        float log_det_term = logf(determinant(cov_q) / determinant(cov_p));
        float kl_divergence = 0.5f * (trace_term + quadratic_term - 3 + log_det_term);
        return kl_divergence;
    }

    __device__ __host__ inline mat3
    conditionCovOrig(const mat3 &cov, vec3 &outEvalues, float epsilon = 1e-10f) {
        mat3 newCov;
        float abseps = fabsf(epsilon);

        // condition diagonal elements
        newCov[0][0] = fmaxf(cov[0][0], abseps);
        newCov[1][1] = fmaxf(cov[1][1], abseps);
        newCov[2][2] = fmaxf(cov[2][2], abseps);

        // condition off diagonal elements
        float sx = sqrtf(cov[0][0]);
        float sy = sqrtf(cov[1][1]);
        float sz = sqrtf(cov[2][2]);

        bool conditioned = false;
        for (float rho = 0.99f; rho >= 0; rho -= 0.01f) {
            float rxy = rho * sx * sy;
            float rxz = rho * sx * sz;
            float ryz = rho * sy * sz;
            newCov[1][0] = clamp(cov[1][0], -rxy, rxy);
            newCov[2][0] = clamp(cov[2][0], -rxz, rxz);
            newCov[2][1] = clamp(cov[2][1], -ryz, ryz);

            newCov[0][1] = newCov[1][0];
            newCov[0][2] = newCov[2][0];
            newCov[1][2] = newCov[2][1];

            // Check
            mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);
            if (outEvalues[0] > 0.0f && outEvalues[1] > 0.0f && outEvalues[2] > 0.0f) {
                conditioned = true;
                break;
            }
        }

        // Check
        if (!conditioned) {
            // Add a small multiple of the identity matrix as a fallback
            newCov = newCov + mat3(1.0f) * abseps;

            // Recompute eigenvalues for the final conditioned matrix
            mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);

            // Check and warn
            if (outEvalues[0] <= 0.0f || outEvalues[1] <= 0.0f || outEvalues[2] <= 0.0f) {
                printf("Warning: cov still non-psd despite conditioning! det: %f\n", determinant(newCov));
                printf("evals: %f, %f, %f\n", outEvalues[0], outEvalues[1], outEvalues[2]);
            }
        }
        return newCov;
    }

    __device__ __host__ inline mat3 conditionCov(const mat3 &cov, vec3 &outEvalues, float epsilon = 1e-10f) {
        mat3 newCov;
        float abseps = fabsf(epsilon);

        // Condition diagonal elements
        newCov[0][0] = fmaxf(cov[0][0], abseps);
        newCov[1][1] = fmaxf(cov[1][1], abseps);
        newCov[2][2] = fmaxf(cov[2][2], abseps);

        // Compute square roots of diagonal elements
        float sx = sqrtf(newCov[0][0]);
        float sy = sqrtf(newCov[1][1]);
        float sz = sqrtf(newCov[2][2]);

        // Condition off-diagonal elements with an adaptive approach
        bool conditioned = false;
        float rho = 0.99f;

        while (rho > 0.0f) {
            float rxy = rho * sx * sy;
            float rxz = rho * sx * sz;
            float ryz = rho * sy * sz;

            newCov[1][0] = clamp(cov[1][0], -rxy, rxy);
            newCov[2][0] = clamp(cov[2][0], -rxz, rxz);
            newCov[2][1] = clamp(cov[2][1], -ryz, ryz);

            newCov[0][1] = newCov[1][0];
            newCov[0][2] = newCov[2][0];
            newCov[1][2] = newCov[2][1];

            // Check if the matrix is PSD using Cholesky decomposition
            if (is_psd(newCov)) {
                conditioned = true;
                break;
            }

            rho -= 0.01f;
        }

        if (!conditioned) {
            // Add a small multiple of the identity matrix as a fallback
            newCov = newCov + mat3(1) * abseps;

            // Recompute eigenvalues for the final conditioned matrix
            mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);

            // Warning if still non-psd
            if (outEvalues[0] <= 0.0f || outEvalues[1] <= 0.0f || outEvalues[2] <= 0.0f) {
                printf("Warning: cov still non-psd despite conditioning! det: %f\n", determinant(newCov));
                printf("evals: %f, %f, %f\n", outEvalues[0], outEvalues[1], outEvalues[2]);
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
