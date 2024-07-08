//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATVEC_H
#define ENGINE24_MATVEC_H

#include "Eigen/Core"
#include "Exceptions.h"

namespace Bcg {
    using Scalar = float;

    template<typename T, int N>
    using Vector = Eigen::Vector<T, N>;

    template<typename T, int M, int N>
    using Matrix = Eigen::Matrix<T, M, N>;

    //! compute perpendicular vector (rotate vector counter-clockwise by 90 degrees)
    template<typename Scalar>
    inline Vector<Scalar, 2> perp(const Vector<Scalar, 2> &v) {
        return Vector<Scalar, 2>(-v[1], v[0]);
    }

    template<typename Scalar>
    inline Vector<Scalar, 3> cross(const Vector<Scalar, 3> &v0,
                                   const Vector<Scalar, 3> &v1) {
        return Vector<Scalar, 3>(v0[1] * v1[2] - v0[2] * v1[1],
                                 v0[2] * v1[0] - v0[0] * v1[2],
                                 v0[0] * v1[1] - v0[1] * v1[0]);
    }

    //! return the inverse of a 4x4 matrix
    template<typename Scalar>
    Matrix<Scalar, 4, 4> inverse(const Matrix<Scalar, 4, 4> &m) {
        Scalar Coef00 = m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2);
        Scalar Coef02 = m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1);
        Scalar Coef03 = m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1);

        Scalar Coef04 = m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2);
        Scalar Coef06 = m(1, 1) * m(3, 3) - m(1, 3) * m(3, 1);
        Scalar Coef07 = m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1);

        Scalar Coef08 = m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2);
        Scalar Coef10 = m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1);
        Scalar Coef11 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);

        Scalar Coef12 = m(0, 2) * m(3, 3) - m(0, 3) * m(3, 2);
        Scalar Coef14 = m(0, 1) * m(3, 3) - m(0, 3) * m(3, 1);
        Scalar Coef15 = m(0, 1) * m(3, 2) - m(0, 2) * m(3, 1);

        Scalar Coef16 = m(0, 2) * m(2, 3) - m(0, 3) * m(2, 2);
        Scalar Coef18 = m(0, 1) * m(2, 3) - m(0, 3) * m(2, 1);
        Scalar Coef19 = m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1);

        Scalar Coef20 = m(0, 2) * m(1, 3) - m(0, 3) * m(1, 2);
        Scalar Coef22 = m(0, 1) * m(1, 3) - m(0, 3) * m(1, 1);
        Scalar Coef23 = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);

        Vector<Scalar, 4> const SignA(+1, -1, +1, -1);
        Vector<Scalar, 4> const SignB(-1, +1, -1, +1);

        Vector<Scalar, 4> Fac0(Coef00, Coef00, Coef02, Coef03);
        Vector<Scalar, 4> Fac1(Coef04, Coef04, Coef06, Coef07);
        Vector<Scalar, 4> Fac2(Coef08, Coef08, Coef10, Coef11);
        Vector<Scalar, 4> Fac3(Coef12, Coef12, Coef14, Coef15);
        Vector<Scalar, 4> Fac4(Coef16, Coef16, Coef18, Coef19);
        Vector<Scalar, 4> Fac5(Coef20, Coef20, Coef22, Coef23);

        Vector<Scalar, 4> Vec0(m(0, 1), m(0, 0), m(0, 0), m(0, 0));
        Vector<Scalar, 4> Vec1(m(1, 1), m(1, 0), m(1, 0), m(1, 0));
        Vector<Scalar, 4> Vec2(m(2, 1), m(2, 0), m(2, 0), m(2, 0));
        Vector<Scalar, 4> Vec3(m(3, 1), m(3, 0), m(3, 0), m(3, 0));

        // clang-format off
        Vector<Scalar, 4> Inv0 = cmult(SignA, (cmult(Vec1, Fac0) - cmult(Vec2, Fac1) + cmult(Vec3, Fac2)));
        Vector<Scalar, 4> Inv1 = cmult(SignB, (cmult(Vec0, Fac0) - cmult(Vec2, Fac3) + cmult(Vec3, Fac4)));
        Vector<Scalar, 4> Inv2 = cmult(SignA, (cmult(Vec0, Fac1) - cmult(Vec1, Fac3) + cmult(Vec3, Fac5)));
        Vector<Scalar, 4> Inv3 = cmult(SignB, (cmult(Vec0, Fac2) - cmult(Vec1, Fac4) + cmult(Vec2, Fac5)));
        // clang-format on

        Matrix<Scalar, 4, 4> Inverse(Inv0, Inv1, Inv2, Inv3);

        Vector<Scalar, 4> Row0(Inverse(0, 0), Inverse(1, 0), Inverse(2, 0),
                               Inverse(3, 0));
        Vector<Scalar, 4> Col0(m(0, 0), m(0, 1), m(0, 2), m(0, 3));

        Scalar Determinant = dot(Col0, Row0);

        Inverse /= Determinant;

        return Inverse;
    }

    template<typename Scalar, int N>
    inline Scalar dot(const Vector<Scalar, N> &v0, const Vector<Scalar, N> &v1) {
        return v0.transpose() * v1;
    }

//! return determinant of 3x3 matrix
    template<typename Scalar>
    Scalar determinant(const Matrix<Scalar, 3, 3> &m) {
        return m(0, 0) * m(1, 1) * m(2, 2) - m(0, 0) * m(1, 2) * m(2, 1) +
               m(1, 0) * m(0, 2) * m(2, 1) - m(1, 0) * m(0, 1) * m(2, 2) +
               m(2, 0) * m(0, 1) * m(1, 2) - m(2, 0) * m(0, 2) * m(1, 1);
    }

//! return the inverse of a 3x3 matrix
    template<typename Scalar>
    Matrix<Scalar, 3, 3> inverse(const Matrix<Scalar, 3, 3> &m) {
        const Scalar det = determinant(m);
        if (fabs(det) < 1.0e-10 || std::isnan(det)) {
            throw SolverException("3x3 matrix not invertible");
        }

        Matrix<Scalar, 3, 3> inv;
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

    //! OpenGL matrix for translation by vector t
    template<typename Scalar>
    Matrix<Scalar, 4, 4> translation_matrix(const Vector<Scalar, 3> &t) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        m(0, 0) = m(1, 1) = m(2, 2) = m(3, 3) = 1.0f;
        m(0, 3) = t[0];
        m(1, 3) = t[1];
        m(2, 3) = t[2];

        return m;
    }

//! OpenGL matrix for scaling x/y/z by s
    template<typename Scalar>
    Matrix<Scalar, 4, 4> scaling_matrix(const Scalar s) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        m(0, 0) = m(1, 1) = m(2, 2) = s;
        m(3, 3) = 1.0f;

        return m;
    }

//! OpenGL matrix for scaling x/y/z by the components of s
    template<typename Scalar>
    Matrix<Scalar, 4, 4> scaling_matrix(const Vector<Scalar, 3> &s) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        m(0, 0) = s[0];
        m(1, 1) = s[1];
        m(2, 2) = s[2];
        m(3, 3) = 1.0f;

        return m;
    }

//! OpenGL matrix for rotation around x-axis by given angle (in degrees)
    template<typename Scalar>
    Matrix<Scalar, 4, 4> rotation_matrix_x(Scalar angle) {
        Scalar ca = cos(angle * Scalar(std::numbers::pi / 180.0));
        Scalar sa = sin(angle * Scalar(std::numbers::pi / 180.0));

        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        m(0, 0) = 1.0;
        m(1, 1) = ca;
        m(1, 2) = -sa;
        m(2, 2) = ca;
        m(2, 1) = sa;
        m(3, 3) = 1.0;

        return m;
    }

//! OpenGL matrix for rotation around y-axis by given angle (in degrees)
    template<typename Scalar>
    Matrix<Scalar, 4, 4> rotation_matrix_y(Scalar angle) {
        Scalar ca = cos(angle * Scalar(std::numbers::pi / 180.0));
        Scalar sa = sin(angle * Scalar(std::numbers::pi / 180.0));

        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        m(0, 0) = ca;
        m(0, 2) = sa;
        m(1, 1) = 1.0;
        m(2, 0) = -sa;
        m(2, 2) = ca;
        m(3, 3) = 1.0;

        return m;
    }

//! OpenGL matrix for rotation around z-axis by given angle (in degrees)
    template<typename Scalar>
    Matrix<Scalar, 4, 4> rotation_matrix_z(Scalar angle) {
        Scalar ca = cos(angle * Scalar(std::numbers::pi / 180.0));
        Scalar sa = sin(angle * Scalar(std::numbers::pi / 180.0));

        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        m(0, 0) = ca;
        m(0, 1) = -sa;
        m(1, 0) = sa;
        m(1, 1) = ca;
        m(2, 2) = 1.0;
        m(3, 3) = 1.0;

        return m;
    }

//! OpenGL matrix for rotation around given axis by given angle (in degrees)
    template<typename Scalar>
    Matrix<Scalar, 4, 4> rotation_matrix(const Vector<Scalar, 3> &axis, Scalar angle) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        Scalar a = angle * Scalar(std::numbers::pi / 180.0f);
        Scalar c = cosf(a);
        Scalar s = sinf(a);
        Scalar one_m_c = Scalar(1) - c;
        Vector<Scalar, 3> ax = normalize(axis);

        m(0, 0) = ax[0] * ax[0] * one_m_c + c;
        m(0, 1) = ax[0] * ax[1] * one_m_c - ax[2] * s;
        m(0, 2) = ax[0] * ax[2] * one_m_c + ax[1] * s;

        m(1, 0) = ax[1] * ax[0] * one_m_c + ax[2] * s;
        m(1, 1) = ax[1] * ax[1] * one_m_c + c;
        m(1, 2) = ax[1] * ax[2] * one_m_c - ax[0] * s;

        m(2, 0) = ax[2] * ax[0] * one_m_c - ax[1] * s;
        m(2, 1) = ax[2] * ax[1] * one_m_c + ax[0] * s;
        m(2, 2) = ax[2] * ax[2] * one_m_c + c;

        m(3, 3) = 1.0f;

        return m;
    }

//! OpenGL matrix for rotation specified by unit quaternion
    template<typename Scalar>
    Matrix<Scalar, 4, 4> rotation_matrix(const Vector<Scalar, 4> &quat) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());
        Scalar s1(1);
        Scalar s2(2);

        m(0, 0) = s1 - s2 * quat[1] * quat[1] - s2 * quat[2] * quat[2];
        m(1, 0) = s2 * quat[0] * quat[1] + s2 * quat[3] * quat[2];
        m(2, 0) = s2 * quat[0] * quat[2] - s2 * quat[3] * quat[1];

        m(0, 1) = s2 * quat[0] * quat[1] - s2 * quat[3] * quat[2];
        m(1, 1) = s1 - s2 * quat[0] * quat[0] - s2 * quat[2] * quat[2];
        m(2, 1) = s2 * quat[1] * quat[2] + s2 * quat[3] * quat[0];

        m(0, 2) = s2 * quat[0] * quat[2] + s2 * quat[3] * quat[1];
        m(1, 2) = s2 * quat[1] * quat[2] - s2 * quat[3] * quat[0];
        m(2, 2) = s1 - s2 * quat[0] * quat[0] - s2 * quat[1] * quat[1];

        m(3, 3) = 1.0f;

        return m;
    }

//! return upper 3x3 matrix from given 4x4 matrix, corresponding to the
//! linear part of an affine transformation
    template<typename Scalar>
    Matrix<Scalar, 3, 3> linear_part(const Matrix<Scalar, 4, 4> &m) {
        return m.block<3, 3>(0, 0);
    }

//! projective transformation of 3D vector v by a 4x4 matrix m:
//! add 1 as 4th component of v, multiply m*v, divide by 4th component
    template<typename Scalar>
    Vector<Scalar, 3> projective_transform(const Matrix<Scalar, 4, 4> &m,
                                           const Vector<Scalar, 3> &v) {
        Vector<Scalar, 3> result = m * v;
        return result.template head<3>() / result[3];
    }

//! affine transformation of 3D vector v by a 4x4 matrix m:
//! add 1 as 4th component of v, multiply m*v, do NOT divide by 4th component
    template<typename Scalar>
    Vector<Scalar, 3> affine_transform(const Matrix<Scalar, 4, 4> &m,
                                       const Vector<Scalar, 3> &v) {
        return linear_transform(m, v) + m.block<3, 1>(0, 3);
    }

//! linear transformation of 3D vector v by a 4x4 matrix m:
//! transform vector by upper-left 3x3 submatrix of m
    template<typename Scalar>
    Vector<Scalar, 3> linear_transform(const Matrix<Scalar, 4, 4> &m,
                                       const Vector<Scalar, 3> &v) {
        return m.block<3, 3>(0, 0) * v;
    }
}

#endif //ENGINE24_MATVEC_H
