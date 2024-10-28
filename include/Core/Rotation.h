//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_ROTATION_H
#define ENGINE24_ROTATION_H

#include "Types.h"
#include "MatUtils.h"

namespace Bcg {
    using Quaternion = Eigen::Quaternion<float>;

    template<typename Scalar>
    Matrix<Scalar, 3, 3> SkewSymmetricMatrix(const Vector<Scalar, 3> &vector) {
        Matrix<Scalar, 3, 3> matrix(0);
        matrix[0][1] = -vector[2];
        matrix[0][2] = vector[1];
        matrix[1][0] = vector[2];
        matrix[1][2] = -vector[0];
        matrix[2][0] = -vector[1];
        matrix[2][1] = vector[0];
        return matrix;
    }

    struct Rotation {
        virtual Matrix<float, 3, 3> matrix() const = 0;

        operator Matrix<float, 3, 3>() const {
            return matrix();
        }

        operator Matrix<float, 4, 4>() const {
            Matrix<float, 4, 4> R = Matrix<float, 4, 4>(1.0f);
            R[0] = Vector<float, 4>(matrix()[0], 1.0f);
            R[1] = Vector<float, 4>(matrix()[1], 1.0f);
            R[2] = Vector<float, 4>(matrix()[2], 1.0f);
            return R;
        }
    };

    struct RotationMatrix : public Rotation {
        RotationMatrix(const Matrix<float, 3, 3> &rot) : rot(rot) {

        }

        Matrix<float, 3, 3> matrix() const override {
            return rot;
        }

        Matrix<float, 3, 3> rot;
    };

    struct AngleAxis : public Rotation {
        AngleAxis(const Matrix<float, 3, 3> &m) {
            float tr = trace(m);
            float angle = std::acos((tr - 1) / 2);
            Vector<float, 3> axis(m[1][2] - m[2][1],
                                  m[2][0] - m[0][2],
                                  m[0][1] - m[1][0]);
            params = SafeNormalize(axis, glm::length(axis)) * angle;
        }

        AngleAxis(const Vector<float, 3> &angle_axis) : params(angle_axis) {

        }

        AngleAxis(float angle, const Vector<float, 3> &angle_axis) {
            params = SafeNormalize(angle_axis, glm::length(angle_axis)) * angle;
        }

        Matrix<float, 3, 3> exp_map(const Vector<float, 3> &r) const {
            //Rodrigues formula
            float angle = glm::length(params);
            if (angle < 1e-6) {
                // For very small angles, use the first-order Taylor expansion
                return Matrix<float, 3, 3>(1.0f) + SkewSymmetricMatrix(r);
            }
            Matrix<float, 3, 3> K = SkewSymmetricMatrix(SafeNormalize(params, angle));
            return Matrix<float, 3, 3>(1.0f) + K * std::sin(angle) + K * K * (1 - std::cos(angle));
        }

        Vector<float, 3> log_map(const Matrix<float, 3, 3> &rot) {
            float tr = trace(rot);
            float angle = std::acos((tr - 1) / 2);
            Vector<float, 3> axis(rot[1][2] - rot[2][1],
                                  rot[2][0] - rot[0][2],
                                  rot[0][1] - rot[1][0]);
            if (angle < 1e-6) {
                // For very small angles, use the first-order approximation
                return axis / 2.0f;
            }
            return SafeNormalize(axis, glm::length(axis)) * angle;
        }

        Matrix<float, 3, 3> matrix() const override {
            return exp_map(params);
        }

        Vector<float, 3> params;
    };

    struct Cayley : public Rotation {
        Cayley(const Matrix<float, 3, 3> &rot) {
            params = Vector<float, 3>(rot[1][2] - rot[2][1],
                                      rot[2][0] - rot[0][2],
                                      rot[0][1] - rot[1][0]);
            params /= 2;
        }

        Cayley(const Vector<float, 3> cayley) : params(cayley) {

        }

        Matrix<float, 3, 3> matrix() const override {
            Matrix<float, 3, 3> K = SkewSymmetricMatrix(params);
            Matrix<float, 3, 3> Id = Matrix<float, 3, 3>(1.0f);
            return (Id - K) * glm::inverse(Id + K);
        }

        Vector<float, 3> params;
    };

    struct TwoAxis : public Rotation {
        TwoAxis(const Matrix<float, 3, 3> &rot) : TwoAxis(rot[0], rot[1]) {

        }

        TwoAxis(const Matrix<float, 2, 3> &two_axis) : TwoAxis(two_axis[0], two_axis[1]) {

        }

        TwoAxis(const Vector<float, 3> &axis1, const Vector<float, 3> &axis2) {
            params[0] = SafeNormalize(axis1, glm::length(axis1));
            params[1] = axis2 - (glm::dot(params[0], axis2)) * params[0];
            params[1] = SafeNormalize(params[1], glm::length(params[1]));
        }

        Matrix<float, 3, 3> matrix() const override {
            return {params[0], params[1], cross(params[0], params[1])};
        }

        Matrix<ScalarType, 2, 3> params;
    };
}

#endif //ENGINE24_ROTATION_H
