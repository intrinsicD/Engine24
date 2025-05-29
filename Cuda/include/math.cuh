//
// Created by alex on 18.08.24.
//

#ifndef ENGINE24_MATH_CUH
#define ENGINE24_MATH_CUH

#include "mat_vec.cuh"
#include <math_constants.h>

namespace Bcg::cuda {
    //Paper: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices
    //for real symmetric 3x3 matrices
    __device__ __host__
    inline float clampf(float f, float min, float max) { return fmaxf(min, fminf(f, max)); }

    __device__ __host__
    inline float clamp(float f, float min, float max) { return fmaxf(min, fminf(f, max)); }

    __device__ __host__
    inline void sort_ascending(vec3 &evals, mat3 &evecs) {
        if (evals[0] > evals[1]) {
            thrust::swap(evals[0], evals[1]);
            thrust::swap(evecs[0], evecs[1]);
        }
        if (evals[0] > evals[2]) {
            thrust::swap(evals[0], evals[2]);
            thrust::swap(evecs[0], evecs[2]);
        }
        if (evals[1] > evals[2]) {
            thrust::swap(evals[1], evals[2]);
            thrust::swap(evecs[1], evecs[2]);
        }
        assert(evals[0] <= evals[1]);
        assert(evals[1] <= evals[2]);
    }

    __host__ __device__
    void rotate(mat3 &m, mat3 &evecs, int p, int q) {
        if (fabs(m[p][q]) > 1e-9f) {  // Only rotate if the off-diagonal element is significant
            float tau = (m[q][q] - m[p][p]) / (2.0f * m[p][q]);
            float t = (tau >= 0.0f ? 1.0f : -1.0f) / (fabs(tau) + sqrt(1.0f + tau * tau));
            float c = 1.0f / sqrt(1.0f + t * t);
            float s = t * c;

            float m_pp = m[p][p];
            float m_qq = m[q][q];

            m[p][p] = c * c * m_pp - 2.0f * c * s * m[p][q] + s * s * m_qq;
            m[q][q] = s * s * m_pp + 2.0f * c * s * m[p][q] + c * c * m_qq;
            m[p][q] = 0.0f;
            m[q][p] = 0.0f;

            for (int i = 0; i < 3; ++i) {
                if (i != p && i != q) {
                    float m_ip = m[i][p];
                    float m_iq = m[i][q];
                    m[i][p] = c * m_ip - s * m_iq;
                    m[p][i] = m[i][p];
                    m[i][q] = c * m_iq + s * m_ip;
                    m[q][i] = m[i][q];
                }
            }

            for (int i = 0; i < 3; ++i) {
                float evec_p = evecs[i][p];
                float evec_q = evecs[i][q];
                evecs[i][p] = c * evec_p - s * evec_q;
                evecs[i][q] = c * evec_q + s * evec_p;
            }
        }
    }


    // Simple Gram-Schmidt orthogonalization
    __host__ __device__ void orthogonalize(mat3 &evecs) {
        evecs[1] = evecs[1] - evecs[0] * (dot(evecs[1], evecs[0]));
        evecs[1] = normalize(evecs[1]);
        evecs[2] = evecs[2] - evecs[0] * (dot(evecs[2], evecs[0]));
        evecs[2] = evecs[2] - evecs[1] * (dot(evecs[2], evecs[1]));
        evecs[2] = normalize(evecs[2]);
    }

    __host__ __device__
    vec3 jacobi_eigen(const mat3 &m_, mat3 &evecs) {
        evecs = mat3(1.0f); // Identity matrix for initial eigenvectors
        float scale = fmax(fmax(fabs(m_[0][0]), fabs(m_[1][1])), fabs(m_[2][2]));
        mat3 m = m_ * (1.0f / scale);

        for (int iter = 0; iter < 50; ++iter) {
            // Find the largest off-diagonal element in the upper triangle
            int p = 0, q = 1;
            float max_offdiag = fabs(m[0][1]);

            if (fabs(m[1][2]) > max_offdiag) {
                p = 1;
                q = 2;
                max_offdiag = fabs(m[1][2]);
            }

            if (fabs(m[0][2]) > max_offdiag) {
                p = 0;
                q = 2;
            }

            // Perform Jacobi rotation
            rotate(m, evecs, p, q);

            // Convergence check: norm of off-diagonal elements
            float offdiag_norm = sqrt(m[0][1] * m[0][1] + m[1][2] * m[1][2] + m[0][2] * m[0][2]);
            float diag_norm = sqrt(m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2]);

            if (offdiag_norm / diag_norm < 1e-9) {
                break;
            }
        }

        // Normalize eigenvectors
        evecs[0] = normalize(evecs[0]);
        evecs[1] = normalize(evecs[1]);
        evecs[2] = normalize(evecs[2]);

        // Orthogonalize (if necessary)
        orthogonalize(evecs);

#ifdef DEBUG_CUDA_BCG
        float ortho_test_01 = fabsf(dot(evecs[0], evecs[1]));
    float ortho_test_12 = fabsf(dot(evecs[1], evecs[2]));
    float ortho_test_20 = fabsf(dot(evecs[2], evecs[0]));

    if (ortho_test_01 > 1e-6f) {
        printf("ortho_test_01: %f\n", ortho_test_01);
    }
    if (ortho_test_12 > 1e-6f) {
        printf("ortho_test_12: %f\n", ortho_test_12);
    }
    if (ortho_test_20 > 1e-6f) {
        printf("ortho_test_20: %f\n", ortho_test_20);
    }
#endif

        // Sort eigenvalues and eigenvectors
        vec3 evals(m[0][0], m[1][1], m[2][2]);
        sort_ascending(evals, evecs);

        return evals * scale;
    }

    __device__ __host__ inline vec3 real_symmetric_3x3_eigendecomposition(const mat3 &m, mat3 &evecs) {
        float a = m[0][0];
        float b = m[1][1];
        float c = m[2][2];
        float d = m[1][0];
        float e = m[2][1];
        float f = m[2][0];
        float dd = d * d;
        float ff = f * f;
        float ee = e * e;
        float abc2 = 2 * a - b - c;
        float bac2 = 2 * b - a - c;
        float cab2 = 2 * c - a - b;
        float x1 = a * a + b * b + c * c - a * b - a * c - b * c + 3 * (dd + ff + ee);
        float x2 = -abc2 * bac2 * cab2 + 9 * (abc2 * dd + bac2 * ff + cab2 * ee) - 54 * d * e * f;

        float phi;
        if (x2 > 0) {
            phi = atan2(sqrt(4 * x1 * x1 * x1 - x2 * x2) / x2, x2);
        } else if (x2 == 0) {
            phi = CUDART_PI / 2;
        } else {
            phi = atan2(sqrt(4 * x1 * x1 * x1 - x2 * x2) / x2, x2) + CUDART_PI;
        }


        float evals_x = (a + b + c - 2 * sqrt(x1) * cos(phi / 3)) / 3;
        float evals_y = (a + b + c + 2 * sqrt(x1) * cos((phi - CUDART_PI) / 3)) / 3;
        float evals_z = (a + b + c + 2 * sqrt(x1) * cos((phi + CUDART_PI) / 3)) / 3;

        float ef = e * f;
        float de = d * e;

        float m1 = (d * (c - evals_x) - ef) / (f * (b - evals_x) - de);
        float m2 = (d * (c - evals_y) - ef) / (f * (b - evals_y) - de);
        float m3 = (d * (c - evals_z) - ef) / (f * (b - evals_z) - de);

        evecs[0] = normalize(vec3((evals_x - c - e * m1) / f, m1, 1));
        evecs[1] = normalize(vec3((evals_y - c - e * m2) / f, m2, 1));
        evecs[2] = normalize(vec3((evals_z - c - e * m3) / f, m3, 1));

        orthogonalize(evecs);


#ifdef DEBUG_CUDA_BCG
        float ortho_test_01 = fabsf(evecs[0].dot(evecs[1]));
        float ortho_test_12 = fabsf(evecs[1].dot(evecs[2]));
        float ortho_test_20 = fabsf(evecs[2].dot(evecs[0]));

        if (ortho_test_01 > 1e-6f) {
            printf("ortho_test_01: %f\n", ortho_test_01);
        }
        if (ortho_test_12 > 1e-6f) {
            printf("ortho_test_12: %f\n", ortho_test_12);
        }
        if (ortho_test_20 > 1e-6f) {
            printf("ortho_test_20: %f\n", ortho_test_20);
        }
#endif

        //sort eigenvalues and eigenvectors
        vec3 evals = vec3(evals_x, evals_y, evals_z);
        sort_ascending(evals, evecs);
        return evals;
    }

    __device__ __host__ inline mat3 cholesky_decomposition(const mat3 &matrix) {
        mat3 L = mat3(0.0f); // Initialize L to zero
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) {
                    L[i][j] = sqrtf(matrix[i][i] - sum);
                } else {
                    L[i][j] = (matrix[i][j] - sum) / L[j][j];
                }
            }
        }
        return L;
    }

    __device__ __host__ inline bool is_psd(const mat3 &matrix) {
        mat3 L = cholesky_decomposition(matrix);

        if (L[0][0] <= 0.0f || L[1][1] <= 0.0f || L[2][2] <= 0.0f) {
            return false; // Matrix is not positive semi-definite
        }
        // If we reach here, the matrix is positive semi-definite
        return true;
    }
}

#endif //ENGINE24_MATH_CUH
