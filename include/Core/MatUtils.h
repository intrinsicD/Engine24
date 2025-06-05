//
// Created by alex on 25.10.24.
//

#ifndef ENGINE24_MATUTILS_H
#define ENGINE24_MATUTILS_H

#include "MatVec.h"

namespace Bcg {
    template<typename T, int N>
    T trace(const Matrix<T, N, N> &m) {
        T sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += m[i][i];
        }
        return sum;
    }

    template<typename T, int N>
    T norm(const Vector<T, N> &v) {
        T sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += v[i] * v[i];
        }
        return sqrtf(sum);
    }

    template<typename T>
    Vector<T, 3> cross(const Vector<T, 3> &u, const Vector<T, 3> &v) {
        return Vector<T, 3>(u[1] * v[2] - u[2] * v[1],
                            u[2] * v[0] - u[0] * v[2],
                            u[0] * v[1] - u[1] * v[0]);
    }

    template<typename T>
    Matrix<T, 4, 4> inverse(const Matrix<T, 4, 4> &m) {
        T Coef00 = m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2);
        T Coef02 = m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1);
        T Coef03 = m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1);

        T Coef04 = m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2);
        T Coef06 = m(1, 1) * m(3, 3) - m(1, 3) * m(3, 1);
        T Coef07 = m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1);

        T Coef08 = m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2);
        T Coef10 = m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1);
        T Coef11 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);

        T Coef12 = m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2);
        T Coef14 = m(0, 1) * m(3, 3) - m(0, 3) * m(3, 1);
        T Coef15 = m(0, 1) * m(3, 2) - m(0, 2) * m(3, 1);

        T Coef16 = m(0, 2) * m(2, 3) - m(0, 3) * m(2, 2);
        T Coef18 = m(0, 1) * m(2, 3) - m(0, 3) * m(2, 1);
        T Coef19 = m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1);

        T Coef20 = m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2);
        T Coef22 = m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1);
        T Coef23 = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);

        Vector<T, 4> const SignA(+1, -1, +1, -1);
        Vector<T, 4> const SignB(-1, +1, -1, +1);

        Vector<T, 4> Fac0(Coef00, Coef00, Coef02, Coef03);
        Vector<T, 4> Fac1(Coef04, Coef04, Coef06, Coef07);
        Vector<T, 4> Fac2(Coef08, Coef08, Coef10, Coef11);
        Vector<T, 4> Fac3(Coef12, Coef12, Coef14, Coef15);
        Vector<T, 4> Fac4(Coef16, Coef16, Coef18, Coef19);
        Vector<T, 4> Fac5(Coef20, Coef20, Coef22, Coef23);

        Vector<T, 4> Vec0(m(0, 1), m(0, 0), m(0, 0), m(0, 0));
        Vector<T, 4> Vec1(m(1, 1), m(1, 0), m(1, 0), m(1, 0));
        Vector<T, 4> Vec2(m(2, 1), m(2, 0), m(2, 0), m(2, 0));
        Vector<T, 4> Vec3(m(3, 1), m(3, 0), m(3, 0), m(3, 0));

        // clang-format off
        Vector<T, 4> Inv0 = cmult(SignA, (cmult(Vec1, Fac0) - cmult(Vec2, Fac1) + cmult(Vec3, Fac2)));
        Vector<T, 4> Inv1 = cmult(SignB, (cmult(Vec0, Fac0) - cmult(Vec2, Fac3) + cmult(Vec3, Fac4)));
        Vector<T, 4> Inv2 = cmult(SignA, (cmult(Vec0, Fac1) - cmult(Vec1, Fac3) + cmult(Vec3, Fac5)));
        Vector<T, 4> Inv3 = cmult(SignB, (cmult(Vec0, Fac2) - cmult(Vec1, Fac4) + cmult(Vec2, Fac5)));
        // clang-format on

        Matrix<T, 4, 4> Inverse(Inv0, Inv1, Inv2, Inv3);

        Vector<T, 4> Row0(Inverse(0, 0), Inverse(1, 0), Inverse(2, 0), Inverse(3, 0));
        Vector<T, 4> Col0(m(0, 0), m(0, 1), m(0, 2), m(0, 3));

        T Determinant = dot(Col0, Row0);

        Inverse /= Determinant;

        return Inverse;
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T geroshin_radius(const glm::mat<3, 3, T, Q> &m) {
        float radius_col0 = m[0].x + fabsf(m[0].y) + fabsf(m[0].z);
        float radius_col1 = m[1].y + fabsf(m[1].x) + fabsf(m[1].z);
        float radius_col2 = m[2].z + fabsf(m[2].x) + fabsf(m[2].y);
        return fmaxf(radius_col0, fmaxf(radius_col1, radius_col2));
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T norm(const glm::mat<3, 3, T, Q> &m) {
        return sqrtf(glm::dot(m[0], m[0]) + glm::dot(m[1], m[1]) + glm::dot(m[2], m[2]));
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T max_row_sum(const glm::mat<3, 3, T, Q> &m) {
        float row0_sum = fabsf(m[0].x) + fabsf(m[0].y) + fabsf(m[0].z);
        float row1_sum = fabsf(m[1].x) + fabsf(m[1].y) + fabsf(m[1].z);
        float row2_sum = fabsf(m[2].x) + fabsf(m[2].y) + fabsf(m[2].z);
        return fmaxf(row0_sum, fmaxf(row1_sum, row2_sum));
    }
}
#endif //ENGINE24_MATUTILS_H
