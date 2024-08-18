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
    __device__ __host__ vec3 real_symmetric_3x3_eigendecomposition(const mat3 &m, mat3 *evecs) {
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
            evec.col0 = vec3((evals.x - c - e * m1) / f, m1, 1);
            evec.col1 = vec3((evals.y - c - e * m2) / f, m2, 1);
            evec.col2 = vec3((evals.z - c - e * m3) / f, m3, 1);

            assert(fabsf(evec.col0.dot(evec.col1)) <= 1e-6f);
            assert(fabsf(evec.col1.dot(evec.col2)) <= 1e-6f);
            assert(fabsf(evec.col2.dot(evec.col0)) <= 1e-6f);
        }
        return evals;
    }
}

#endif //ENGINE24_MATH_CUH
