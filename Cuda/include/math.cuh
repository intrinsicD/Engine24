//
// Created by alex on 18.08.24.
//

#ifndef ENGINE24_MATH_CUH
#define ENGINE24_MATH_CUH

#include "vec3.cuh"
#include "mat3.cuh"
#include <math_constants.h>

namespace Bcg::cuda {
    //Paper: Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices
    //for real symmetric 3x3 matrices
    __device__ __host__
    inline float clampf(float f, float min, float max) { return fmaxf(min, fminf(f, max)); }

    __device__ __host__
    inline double clamp(double f, double min, double max) { return fmax(min, fmin(f, max)); }

    __device__ __host__
    inline void sort_ascending(vec3 &evals, mat3 &evecs) {
        if (evals.x > evals.y) {
            thrust::swap(evals.x, evals.y);
            thrust::swap(evecs.col0, evecs.col1);
        }
        if (evals.x > evals.z) {
            thrust::swap(evals.x, evals.z);
            thrust::swap(evecs.col0, evecs.col2);
        }
        if (evals.y > evals.z) {
            thrust::swap(evals.y, evals.z);
            thrust::swap(evecs.col1, evecs.col2);
        }
        assert(evals.x <= evals.y);
        assert(evals.y <= evals.z);
    }

    __host__ __device__
    void rotate(mat3 &m, mat3 &evecs, int p, int q) {
        if (fabs(m[p][q]) > 1e-9) {  // Only rotate if the off-diagonal element is significant
            double tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
            double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;

            double m_pp = m[p][p];
            double m_qq = m[q][q];

            m[p][p] = c * c * m_pp - 2.0 * c * s * m[p][q] + s * s * m_qq;
            m[q][q] = s * s * m_pp + 2.0 * c * s * m[p][q] + c * c * m_qq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;

            for (int i = 0; i < 3; ++i) {
                if (i != p && i != q) {
                    double m_ip = m[i][p];
                    double m_iq = m[i][q];
                    m[i][p] = c * m_ip - s * m_iq;
                    m[p][i] = m[i][p];
                    m[i][q] = c * m_iq + s * m_ip;
                    m[q][i] = m[i][q];
                }
            }

            for (int i = 0; i < 3; ++i) {
                double evec_p = evecs[i][p];
                double evec_q = evecs[i][q];
                evecs[i][p] = c * evec_p - s * evec_q;
                evecs[i][q] = c * evec_q + s * evec_p;
            }
        }
    }


    // Simple Gram-Schmidt orthogonalization
    __host__ __device__ void orthogonalize(mat3 &evecs) {
        evecs.col1 = evecs.col1 - evecs.col0 * (evecs.col1.dot(evecs.col0));
        evecs.col1 = evecs.col1.normalized();
        evecs.col2 = evecs.col2 - evecs.col0 * (evecs.col2.dot(evecs.col0));
        evecs.col2 = evecs.col2 - evecs.col1 * (evecs.col2.dot(evecs.col1));
        evecs.col2 = evecs.col2.normalized();
    }

    __host__ __device__
    vec3 jacobi_eigen(const mat3 &m_, mat3 &evecs) {
        evecs = mat3::identity();
        double scale = fmax(fmax(fabs(m_.col0.x), fabs(m_.col1.y)), fabs(m_.col2.z));
        mat3 m = m_ * (1.0 / scale);
        for (int iter = 0; iter < 50; ++iter) {
            // Find the largest off-diagonal element in the upper triangle
            int p = 0, q = 1;
            double max_offdiag = fabs(m[0][1]);

            if (fabs(m[1][2]) > max_offdiag) {
                p = 1;
                q = 2;
                max_offdiag = fabs(m[1][2]);
            }

            if (fabs(m[0][2]) > max_offdiag) {
                p = 0;
                q = 2;
            }

            rotate(m, evecs, p, q);

            // Convergence check: norm of off-diagonal elements
            double offdiag_norm = sqrt(m[0][1] * m[0][1] + m[1][2] * m[1][2] + m[0][2] * m[0][2]);
            double diag_norm = sqrt(m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2]);

            if (offdiag_norm / diag_norm < 1e-9) {
                break;
            }
        }
        evecs.col0 = evecs.col0.normalized();
        evecs.col1 = evecs.col1.normalized();
        evecs.col2 = evecs.col2.normalized();

        orthogonalize(evecs);

#ifdef DEBUG_CUDA_BCG
        float ortho_test_01 = fabsf(evecs.col0.dot(evecs.col1));
        float ortho_test_12 = fabsf(evecs.col1.dot(evecs.col2));
        float ortho_test_20 = fabsf(evecs.col2.dot(evecs.col0));

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
        vec3 evals = vec3(m[0][0], m[1][1], m[2][2]);
        sort_ascending(evals, evecs);
        return evals * scale;
    }

    __device__ __host__ inline vec3 real_symmetric_3x3_eigendecomposition(const mat3 &m, mat3 &evecs) {
        double a = m.col0.x;
        double b = m.col1.y;
        double c = m.col2.z;
        double d = m.col1.x;
        double e = m.col2.y;
        double f = m.col2.x;
        double dd = d * d;
        double ff = f * f;
        double ee = e * e;
        double abc2 = 2 * a - b - c;
        double bac2 = 2 * b - a - c;
        double cab2 = 2 * c - a - b;
        double x1 = a * a + b * b + c * c - a * b - a * c - b * c + 3 * (dd + ff + ee);
        double x2 = -abc2 * bac2 * cab2 + 9 * (abc2 * dd + bac2 * ff + cab2 * ee) - 54 * d * e * f;

        double phi;
        if (x2 > 0) {
            phi = atan2(sqrt(4 * x1 * x1 * x1 - x2 * x2) / x2, x2);
        } else if (x2 == 0) {
            phi = CUDART_PI / 2;
        } else {
            phi = atan2(sqrt(4 * x1 * x1 * x1 - x2 * x2) / x2, x2) + CUDART_PI;
        }


        double evals_x = (a + b + c - 2 * sqrt(x1) * cos(phi / 3)) / 3;
        double evals_y = (a + b + c + 2 * sqrt(x1) * cos((phi - CUDART_PI) / 3)) / 3;
        double evals_z = (a + b + c + 2 * sqrt(x1) * cos((phi + CUDART_PI) / 3)) / 3;

        double ef = e * f;
        double de = d * e;

        double m1 = (d * (c - evals_x) - ef) / (f * (b - evals_x) - de);
        double m2 = (d * (c - evals_y) - ef) / (f * (b - evals_y) - de);
        double m3 = (d * (c - evals_z) - ef) / (f * (b - evals_z) - de);

        evecs.col0 = vec3((evals_x - c - e * m1) / f, m1, 1).normalized();
        evecs.col1 = vec3((evals_y - c - e * m2) / f, m2, 1).normalized();
        evecs.col2 = vec3((evals_z - c - e * m3) / f, m3, 1).normalized();

        orthogonalize(evecs);



#ifdef DEBUG_CUDA_BCG
        float ortho_test_01 = fabsf(evecs.col0.dot(evecs.col1));
        float ortho_test_12 = fabsf(evecs.col1.dot(evecs.col2));
        float ortho_test_20 = fabsf(evecs.col2.dot(evecs.col0));

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

    __device__ __host__ inline bool is_psd(const mat3 &matrix) {
        mat3 L; // Lower triangular matrix

        // Cholesky decomposition
        for (int i = 0; i < 3; ++i) {
            // Compute the diagonal element
            double sum = 0.0f;
            for (int k = 0; k < i; ++k) {
                sum += L[k][i] * L[k][i];  // Adjusted for column-major format
            }
            double diag = matrix[i][i] - sum;

            if (diag <= 0.0f) {
                return false; // Matrix is not positive semi-definite
            }

            L[i][i] = sqrtf(diag);

            // Compute the off-diagonal elements
            for (int j = i + 1; j < 3; ++j) {
                sum = 0.0f;
                for (int k = 0; k < i; ++k) {
                    sum += L[k][i] * L[k][j];  // Adjusted for column-major format
                }
                L[i][j] = (matrix[i][j] - sum) / L[i][i];  // Adjusted for column-major format
            }
        }

        return true; // Matrix is positive semi-definite
    }
}

#endif //ENGINE24_MATH_CUH
