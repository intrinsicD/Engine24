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
    __device__ __host__ inline float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }

    __host__ __device__ void rotate(mat3 &m, mat3 &evecs, int p, int q) {
        if (m[p][q] != 0.0f) {
            float tau = (m[q][q] - m[p][p]) / (2.0f * m[p][q]);
            float t = (tau >= 0 ? 1.0f : -1.0f) / (std::abs(tau) + std::sqrt(1.0f + tau * tau));
            float c = 1.0f / std::sqrt(1.0f + t * t);
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

    __host__ __device__ vec3 jacobi_eigen(mat3 &m, mat3 &evecs) {
        evecs = mat3(vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));

        for (int iter = 0; iter < 50; ++iter) {
            int p = 0, q = 1;
            if (std::abs(m[0][1]) < std::abs(m[1][2])) { p = 1; q = 2; }
            if (std::abs(m[p][q]) < std::abs(m[0][2])) { p = 0; q = 2; }

            rotate(m, evecs, p, q);

            if (std::abs(m[0][1]) < 1e-9 && std::abs(m[1][2]) < 1e-9 && std::abs(m[0][2]) < 1e-9) {
                break;
            }
        }
        assert(fabsf(evecs.col0.dot(evecs.col1)) <= 1e-6f);
        assert(fabsf(evecs.col1.dot(evecsm ".col2)) <= 1e-6f);
        assert(fabsf(evecs.col2.dot(evecs.col0)) <= 1e-6f);
        return vec3(m[0][0], m[1][1], m[2][2]);
    }

    __device__ __host__ inline vec3 real_symmetric_3x3_eigendecomposition(const mat3 &m, mat3 *evecs) {
        float a = m.col0.x;
        float b = m.col1.y;
        float c = m.col2.z;
        float d = m.col1.x;
        float e = m.col2.y;
        float f = m.col2.x;
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
            phi = atan2f(sqrtf(4 * x1 * x1 * x1 - x2 * x2) / x2, x2);
        } else if (x2 == 0) {
            phi = CUDART_PI / 2;
        } else {
            phi = atan2f(sqrtf(4 * x1 * x1 * x1 - x2 * x2) / x2, x2) + CUDART_PI;
        }

        vec3 evals;
        evals.x = (a + b + c - 2 * sqrtf(x1) * cosf(phi / 3)) / 3;
        evals.y = (a + b + c + 2 * sqrtf(x1) * cosf((phi - CUDART_PI) / 3)) / 3;
        evals.z = (a + b + c + 2 * sqrtf(x1) * cosf((phi + CUDART_PI) / 3)) / 3;

        if (evecs) {
            float ef = e * f;
            float de = d * e;

            float m1 = (d * (c - evals.x) - ef) / (f * (b - evals.x) - de);
            float m2 = (d * (c - evals.y) - ef) / (f * (b - evals.y) - de);
            float m3 = (d * (c - evals.z) - ef) / (f * (b - evals.z) - de);

            mat3 &evec = *evecs;
            evec.col0 = vec3((evals.x - c - e * m1) / f, m1, 1).normalized();
            evec.col1 = vec3((evals.y - c - e * m2) / f, m2, 1).normalized();
            evec.col2 = vec3((evals.z - c - e * m3) / f, m3, 1).normalized();

            assert(fabsf(evec.col0.dot(evec.col1)) <= 1e-6f);
            assert(fabsf(evec.col1.dot(evec.col2)) <= 1e-6f);
            assert(fabsf(evec.col2.dot(evec.col0)) <= 1e-6f);
        }
        return evals;
    }
}

#endif //ENGINE24_MATH_CUH
