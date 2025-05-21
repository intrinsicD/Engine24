//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_ROTATION_H
#define ENGINE24_ROTATION_H

#include "Types.h"
#include "MatUtils.h"

namespace Bcg {
    template<typename T>
    Matrix<T, 3, 3> SkewSymmetricMatrix(const Vector<T, 3> &vector) {
        Matrix<T, 3, 3> matrix(0);
        matrix[0][1] = -vector[2];
        matrix[0][2] = vector[1];
        matrix[1][0] = vector[2];
        matrix[1][2] = -vector[0];
        matrix[2][0] = -vector[1];
        matrix[2][1] = vector[0];
        return matrix;
    }

    template<typename T>
    class IRotation {
    public:
        virtual Matrix<T, 3, 3> matrix() const = 0;

        operator Matrix<T, 3, 3>() const {
            return matrix();
        }

        operator Matrix<T, 4, 4>() const {
            Matrix<T, 4, 4> R = Matrix<T, 4, 4>(1.0f);
            R[0] = Vector<T, 4>(matrix()[0], 0.0f);
            R[1] = Vector<T, 4>(matrix()[1], 0.0f);
            R[2] = Vector<T, 4>(matrix()[2], 0.0f);
            return R;
        }

        Vector<T, 3> operator*(const Vector<T, 3> &v) const {
            return Matrix<T, 3, 3>(*this) * v;
        }

        Matrix<T, 3, 3> operator*(const Matrix<T, 3, 3> &m) const {
            return Matrix<T, 3, 3>(*this) * m;
        }

        Vector<T, 4> operator*(const Vector<T, 4> &v) const {
            return Matrix<T, 4, 4>(*this) * v;
        }

        Matrix<T, 4, 4> operator*(const Matrix<T, 4, 4> &m) const {
            return Matrix<T, 4, 4>(*this) * m;
        }
    };

    template<typename T>
    class RotationMatrix : public IRotation<T> {
    public:
        RotationMatrix(const Matrix<T, 3, 3> &rot) : m_rot(rot) {

        }

        RotationMatrix(const Matrix<T, 4, 4> &rot) : m_rot(Matrix<T, 3, 3>(rot)) {

        }

        Matrix<T, 3, 3> matrix() const override {
            return m_rot;
        }

    private:
        Matrix<T, 3, 3> m_rot;
    };

    template<typename T>
    class AngleAxis : public IRotation<T> {
    public:
        AngleAxis() : m_axis(0.0, 0.0, 1.0), m_angle(0.0) {

        }

        AngleAxis(const Matrix<T, 3, 3> &m) {
            T tr = trace(m);
            m_angle = std::acos((tr - 1) / 2);
            Vector<T, 3> axis(m[1][2] - m[2][1],
                              m[2][0] - m[0][2],
                              m[0][1] - m[1][0]);
            m_axis = SafeNormalize(axis, glm::length(axis));
        }

        AngleAxis(const Vector<T, 3> &angle_axis) : AngleAxis(glm::length(angle_axis), angle_axis) {

        }

        AngleAxis(T angle, const Vector<T, 3> &axis) : m_angle(angle), m_axis(SafeNormalize(axis, glm::length(axis))) {

        }

        Matrix<T, 3, 3> exp_map() const {
            //Rodrigues formula
            if (m_angle < 1e-6) {
                // For very small angles, use the first-order Taylor expansion
                return Matrix<T, 3, 3>(1.0f) + SkewSymmetricMatrix(m_axis);
            }
            Matrix<T, 3, 3> K = SkewSymmetricMatrix(m_axis);
            return Matrix<T, 3, 3>(1.0f) + K * std::sin(m_angle) + K * K * (1 - std::cos(m_angle));
        }

        void log_map(const Matrix<T, 3, 3> &rot) {
            T tr = trace(rot);
            m_angle = std::acos((tr - 1) / 2);
            Vector<T, 3> axis(rot[1][2] - rot[2][1],
                              rot[2][0] - rot[0][2],
                              rot[0][1] - rot[1][0]);
            if (angle < 1e-6) {
                // For very small angles, use the first-order approximation
                m_axis = axis / 2.0f;
            }
            m_axis = SafeNormalize(axis, glm::length(axis));
        }

        Matrix<T, 3, 3> matrix() const override {
            return exp_map();
        }

        Vector<T, 3> axis() const {
            return m_axis;
        }

        T angle() const {
            return m_angle;
        }

    private:
        Vector<T, 3> m_axis;
        T m_angle;
    };

    template<typename T>
    class Cayley : public IRotation<T> {
    public:
        Cayley() : m_params(0.0, 0.0, 0.0) {

        }

        Cayley(const Vector<T, 3> cayley) : m_params(cayley) {

        }

        Cayley(const Matrix<T, 3, 3> &rot) {
            m_params = Vector<T, 3>(rot[1][2] - rot[2][1],
                                    rot[2][0] - rot[0][2],
                                    rot[0][1] - rot[1][0]);
            m_params /= 2;
        }

        Matrix<T, 3, 3> matrix() const override {
            Matrix<T, 3, 3> K = SkewSymmetricMatrix(m_params);
            Matrix<T, 3, 3> Id = Matrix<T, 3, 3>(1.0f);
            return (Id - K) * glm::inverse(Id + K);
        }

    private:
        Vector<T, 3> m_params;
    };


    template<typename T>
    class TwoAxis : public IRotation<T> {
    public:
        TwoAxis() : TwoAxis(Vector<T, 3>(1.0, 0.0, 0.0), Vector<T, 3>(0.0, 1.0, 0.0)) {

        }

        TwoAxis(const Matrix<T, 3, 3> &rot) : TwoAxis(rot[0], rot[1]) {

        }

        TwoAxis(const Matrix<T, 2, 3> &two_axis) : TwoAxis(two_axis[0], two_axis[1]) {

        }

        TwoAxis(const Vector<T, 3> &axis1, const Vector<T, 3> &axis2) {
            m_params[0] = SafeNormalize(axis1, glm::length(axis1));
            m_params[1] = axis2 - (glm::dot(m_params[0], axis2)) * m_params[0];
            m_params[1] = SafeNormalize(m_params[1], glm::length(m_params[1]));
        }

        Matrix<T, 3, 3> matrix() const override {
            return {m_params[0], m_params[1], cross(m_params[0], m_params[1])};
        }

    private:
        Matrix<ScalarType, 2, 3> m_params;
    };
}

#endif //ENGINE24_ROTATION_H
