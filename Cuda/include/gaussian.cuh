//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_GAUSSIAN_CUH
#define ENGINE24_GAUSSIAN_CUH

#include "glm/glm.hpp"
#include "MatUtils.h"
#include "math.cuh"

namespace Bcg::cuda {
    struct gaussian {
        glm::vec3 mean;
        glm::mat3 cov;
    };

    __device__ __host__ inline float gaussian_pdf(const glm::vec3 &mean, const glm::vec3 &x, float det, const glm::mat3 &inv_cov) {
        glm::vec3 diff = x - mean;
        float exponent = -0.5f * glm::dot(diff, inv_cov * diff);
        return expf(exponent) / sqrtf(det * 8 * CUDART_PI);
    }

    __device__ __host__ inline float
    kullback_leibler_divergence(const glm::vec3 &mean_p, const glm::mat3 &cov_p,
                                const glm::vec3 &mean_q, const glm::mat3 &cov_q, const glm::mat3 &inv_cov_q) {
        float trace_term = trace(inv_cov_q * cov_p);
        glm::vec3 diff = mean_q - mean_p;
        float quadratic_term = glm::dot(diff, inv_cov_q * diff);
        float log_det_term = logf(glm::determinant(cov_q) / glm::determinant(cov_p));
        float kl_divergence = 0.5f * (trace_term + quadratic_term - 3 + log_det_term);
        return kl_divergence;
    }

    __device__ __host__ inline glm::mat3
    conditionCovOrig(const glm::mat3 &cov, glm::vec3 &outEvalues, float epsilon = 1e-10f) {
        glm::mat3 newCov;
        float abseps = fabsf(epsilon);

        // condition diagonal elements
        newCov[0].x = fmaxf(cov[0].x, abseps);
        newCov[1].y = fmaxf(cov[1].y, abseps);
        newCov[2].z = fmaxf(cov[2].z, abseps);

        // condition off diagonal elements
        float sx = sqrtf(cov[0].x);
        float sy = sqrtf(cov[1].y);
        float sz = sqrtf(cov[2].z);

        bool conditioned = false;
        for (float rho = 0.99f; rho >= 0; rho -= 0.01f) {
            float rxy = rho * sx * sy;
            float rxz = rho * sx * sz;
            float ryz = rho * sy * sz;
            newCov[1].x = clamp(cov[1].x, -rxy, rxy);
            newCov[2].x = clamp(cov[2].x, -rxz, rxz);
            newCov[2].y = clamp(cov[2].y, -ryz, ryz);

            newCov[0].y = newCov[1].x;
            newCov[0].z = newCov[2].x;
            newCov[1].z = newCov[2].y;

            // Check
            glm::mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);
            if (outEvalues.x > 0.0f && outEvalues.y > 0.0f && outEvalues.z > 0.0f) {
                conditioned = true;
                break;
            }
        }

        // Check
        if (!conditioned) {
            // Add a small multiple of the identity matrix as a fallback
            newCov = newCov + glm::mat3(1) * abseps;

            // Recompute eigenvalues for the final conditioned matrix
            glm::mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);

            // Check and warn
            if (outEvalues.x <= 0.0f || outEvalues.y <= 0.0f || outEvalues.z <= 0.0f) {
                printf("Warning: cov still non-psd despite conditioning! det: %f\n", glm::determinant(newCov));
                printf("evals: %f, %f, %f\n", outEvalues.x, outEvalues.y, outEvalues.z);
            }
        }
        return newCov;
    }

    __device__ __host__ inline glm::mat3 conditionCov(const glm::mat3 &cov, glm::vec3 &outEvalues, float epsilon = 1e-10f) {
        glm::mat3 newCov;
        float abseps = fabsf(epsilon);

        // Condition diagonal elements
        newCov[0].x = fmaxf(cov[0].x, abseps);
        newCov[1].y = fmaxf(cov[1].y, abseps);
        newCov[2].z = fmaxf(cov[2].z, abseps);

        // Compute square roots of diagonal elements
        float sx = sqrtf(newCov[0].x);
        float sy = sqrtf(newCov[1].y);
        float sz = sqrtf(newCov[2].z);

        // Condition off-diagonal elements with an adaptive approach
        bool conditioned = false;
        float rho = 0.99f;

        while (rho > 0.0f) {
            float rxy = rho * sx * sy;
            float rxz = rho * sx * sz;
            float ryz = rho * sy * sz;

            newCov[1].x = clamp(cov[1].x, -rxy, rxy);
            newCov[2].x = clamp(cov[2].x, -rxz, rxz);
            newCov[2].y = clamp(cov[2].y, -ryz, ryz);

            newCov[0].y = newCov[1].x;
            newCov[0].z = newCov[2].x;
            newCov[1].z = newCov[2].y;

            // Check if the matrix is PSD using Cholesky decomposition
            if (is_psd(newCov)) {
                conditioned = true;
                break;
            }

            rho -= 0.01f;
        }

        if (!conditioned) {
            // Add a small multiple of the identity matrix as a fallback
            newCov = newCov + glm::mat3(1) * abseps;

            // Recompute eigenvalues for the final conditioned matrix
            glm::mat3 evecs;
            outEvalues = jacobi_eigen(newCov, evecs);

            // Warning if still non-psd
            if (outEvalues.x <= 0.0f || outEvalues.y <= 0.0f || outEvalues.z <= 0.0f) {
                printf("Warning: cov still non-psd despite conditioning! det: %f\n", glm::determinant(newCov));
                printf("evals: %f, %f, %f\n", outEvalues.x, outEvalues.y, outEvalues.z);
            }
        }

        return newCov;
    }

    __device__ __host__ inline glm::mat3 conditionCov(const glm::mat3 &cov, float epsilon = 1e-10f) {
        glm::vec3 evd;
        return conditionCov(cov, evd, epsilon);
    }

}

#endif //ENGINE24_GAUSSIAN_CUH
