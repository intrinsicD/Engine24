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
