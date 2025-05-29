//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATVEC_H
#define ENGINE24_MATVEC_H

#include "Eigen/Core"
#include "Exceptions.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"

#include "glm/gtx/component_wise.hpp"
#include "glm/gtc/type_ptr.hpp"

namespace Bcg {
/*    template<typename T, int N>
    using Vector = Eigen::Vector<T, N>;

    template<typename T, int M, int N>
    using Matrix = Eigen::Matrix<T, M, N>;
    */
    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    using Vector = glm::vec<N, T, Q>;

    template<typename T, int C, int R, glm::qualifier Q = glm::defaultp>
    using Matrix = glm::mat<C, R, T, Q>;

    template<typename T, int N>
    Vector<T, N> SafeNormalize(const Vector<T, N> &v, T length, T epsilon = 1e-6) {
        return v / std::max(length, epsilon);
    }

    //! compute perpendicular vector (rotate vector counter-clockwise by 90 degrees)
    template<typename T>
    inline Vector<T, 2> perp(const Vector<T, 2> &v) {
        return Vector<T, 2>(-v[1], v[0]);
    }

    template<typename T>
    inline Vector<T, 3> cross(const Vector<T, 3> &v0, const Vector<T, 3> &v1) {
        return Vector<T, 3>(v0[1] * v1[2] - v0[2] * v1[1],
                            v0[2] * v1[0] - v0[0] * v1[2],
                            v0[0] * v1[1] - v0[1] * v1[0]);
    }

    //! return the inverse of a 4x4 matrix
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

    template<typename T, int N>
    inline T dot(const Vector<T, N> &v0, const Vector<T, N> &v1) {
        return v0.transpose() * v1;
    }

//! return determinant of 3x3 matrix
    template<typename T>
    T determinant(const Matrix<T, 3, 3> &m) {
        return m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(2, 1) +
               m(1, 0) * m(0, 2) * m(2, 1) - m(1, 0) * m(0, 1) * m(2, 2) +
               m(2, 0) * m(0, 1) * m(1, 2) - m(2, 0) * m(0, 2) * m(1, 1);
    }

//! return the inverse of a 3x3 matrix
    template<typename T>
    Matrix<T, 3, 3> inverse(const Matrix<T, 3, 3> &m) {
        const T det = determinant(m);
        if (fabs(det) < 1.0e-10 || std::isnan(det)) {
            throw SolverException("3x3 matrix not invertible");
        }

        Matrix<T, 3, 3> inv;
        inv(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) / det;
        inv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) / det;
        inv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) / det;
        inv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) / det;
        inv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) / det;
        inv(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) / det;
        inv(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) / det;
        inv(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) / det;
        inv(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) / det;

        return inv;
    }

//! return upper 3x3 matrix from given 4x4 matrix, corresponding to the
//! linear part of an affine transformation
    template<typename T>
    Matrix<T, 3, 3> linear_part(const Matrix<T, 4, 4> &m) {
        return m.block(0, 0, 3, 3);
    }

//! projective transformation of 3D vector v by a 4x4 matrix m:
//! add 1 as 4th component of v, multiply m*v, divide by 4th component
    template<typename T>
    Vector<T, 3> projective_transform(const Matrix<T, 4, 4> &m,
                                      const Vector<T, 3> &v) {
        Vector<T, 3> result = m * v;
        return result.template head<3>() / result[3];
    }

//! affine transformation of 3D vector v by a 4x4 matrix m:
//! add 1 as 4th component of v, multiply m*v, do NOT divide by 4th component
    template<typename T>
    Vector<T, 3> affine_transform(const Matrix<T, 4, 4> &m,
                                  const Vector<T, 3> &v) {
        return linear_transform(m, v) + m.block(0, 3, 3, 1);
    }

//! linear transformation of 3D vector v by a 4x4 matrix m:
//! transform vector by upper-left 3x3 submatrix of m
    template<typename T>
    Vector<T, 3> linear_transform(const Matrix<T, 4, 4> &m,
                                  const Vector<T, 3> &v) {
        return m.block(0, 0, 3, 3) * v;
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    random_like(const Eigen::MatrixBase<Derived> &matrix) {
        return Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Random(
                matrix.derived().rows(), matrix.derived().cols());
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    zeros_like(const Eigen::MatrixBase<Derived> &matrix) {
        return Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Zero(
                matrix.derived().rows(), matrix.derived().cols());
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    ones_like(const Eigen::MatrixBase<Derived> &matrix) {
        return Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Ones(
                matrix.derived().rows(), matrix.derived().cols());
    }
}

#endif //ENGINE24_MATVEC_H
