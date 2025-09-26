//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATVEC_H
#define ENGINE24_MATVEC_H

#include "Eigen/Core"
#include "Exceptions.h"
#include "Macros.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"

#include "glm/gtx/component_wise.hpp"
#include "glm/gtx/compatibility.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/quaternion.hpp"

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

    template<typename T, glm::qualifier Q = glm::defaultp>
    using Quaternion = glm::qua<T, Q>;

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

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> operator+(const Vector<T, 3> &a, const Vector<T, 3> &b) {
        return Vector<T, 3>(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> operator-(const Vector<T, 3> &a, const Vector<T, 3> &b) {
        return Vector<T, 3>(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> operator*(T s, const Vector<T, 3> &v) { return Vector<T, 3>(s * v.x, s * v.y, s * v.z); }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> operator*(const Vector<T, 3> &v, T s) { return s * v; }

    template<typename T>
    CUDA_HOST_DEVICE inline T dot(const Vector<T, 3> &a, const Vector<T, 3> &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> hadamard(const Vector<T, 3> &a, const Vector<T, 3> &b) {
        return Vector<T, 3>(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> inv(const Vector<T, 3> &a) { return Vector<T, 3>(T(1) / a.x, T(1) / a.y, T(1) / a.z); }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> cross(const Vector<T, 3> &a, const Vector<T, 3> &b) {
        return Vector<T, 3>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Matrix<T, 3, 3> outer(const Vector<T, 3> &a, const Vector<T, 3> &b) {
        Matrix<T, 3, 3> M;
        M[0][0] = a.x * b.x;
        M[0][1] = a.x * b.y;
        M[0][2] = a.x * b.z;
        M[1][0] = a.y * b.x;
        M[1][1] = a.y * b.y;
        M[1][2] = a.y * b.z;
        M[2][0] = a.z * b.x;
        M[2][1] = a.z * b.y;
        M[2][2] = a.z * b.z;
        return M;
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Matrix<T, 3, 3> add(const Matrix<T, 3, 3> &A, const Matrix<T, 3, 3> &B) {
        Matrix<T, 3, 3> C;
        for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) C[c][r] = A[c][r] + B[c][r];
        return C;
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Matrix<T, 3, 3> sub(const Matrix<T, 3, 3> &A, const Matrix<T, 3, 3> &B) {
        Matrix<T, 3, 3> C;
        for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) C[c][r] = A[c][r] - B[c][r];
        return C;
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Matrix<T, 3, 3> mul(const Matrix<T, 3, 3> &A, T s) {
        Matrix<T, 3, 3> C;
        for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) C[c][r] = A[c][r] * s;
        return C;
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> matvec(const Matrix<T, 3, 3> &A, const Vector<T, 3> &x) {
        return Vector<T, 3>(A[0][0] * x.x + A[0][1] * x.y + A[0][2] * x.z,
                  A[1][0] * x.x + A[1][1] * x.y + A[1][2] * x.z,
                  A[2][0] * x.x + A[2][1] * x.y + A[2][2] * x.z);
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Matrix<T, 3, 3> eye() {
        Matrix<T, 3, 3> I;
        I[0][0] = 1;
        I[0][1] = 0;
        I[0][2] = 0;
        I[1][0] = 0;
        I[1][1] = 1;
        I[1][2] = 0;
        I[2][0] = 0;
        I[2][1] = 0;
        I[2][2] = 1;
        return I;
    }

    template<typename T>
    CUDA_HOST_DEVICE inline T norm(const Vector<T, 3> &a) { return std::sqrt(dot(a, a)); }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> normalize(const Vector<T, 3> &a) {
        T n = norm(a);
        return (n > T(0)) ? a * (T(1) / n) : Vector<T, 3>(T(0), T(0), T(0));
    }

    template<typename T>
    CUDA_HOST_DEVICE inline Vector<T, 3> exp_vec(const Vector<T, 3> &a) {
        return Vector<T, 3>(std::exp(a.x), std::exp(a.y), std::exp(a.z));
    }

    // Rotate v by a **unit** quaternion q: v' = q * v * q^{-1}
    template<typename T>
    CUDA_HOST_DEVICE
    inline Vector<T, 3> rotate_by_unit_quat(const Quaternion<T> &q, const Vector<T, 3> &v) {
        const Vector<T, 3> u(q.x, q.y, q.z);
        const Vector<T, 3> t = T(2) * cross(u, v);
        return v + q.w * t + cross(u, t);
    }

    // Apply R^T (i.e., rotate by the **conjugate**). Use this to go world → local.
    template<typename T>
    CUDA_HOST_DEVICE
    inline Vector<T, 3> rotate_by_unit_quat_conjugate(const Quaternion<T> &q, const Vector<T, 3> &v) {
        const Quaternion<T> qc{q.w, -q.x, -q.y, -q.z};
        return rotate_by_unit_quat(qc, v);
    }
}

#endif //ENGINE24_MATVEC_H
