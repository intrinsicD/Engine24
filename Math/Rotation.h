//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_ROTATION_H
#define ENGINE24_ROTATION_H

#include "MatVec.h"
#include "Eigen/Geometry"

namespace Bcg {
    using Quaternion = Eigen::Quaternion<float>;

    template<typename Scalar>
    Matrix<Scalar, 3, 3> SkewSymmetricMatrix(const Vector<Scalar, 3> &vector) {
        Matrix<Scalar, 3, 3> matrix;
        matrix << 0, -vector[2], vector[1],
                vector[2], 0, -vector[0],
                -vector[1], vector[0], 0;
        return matrix;
    }

    struct Rotation {
        virtual Matrix<float, 3, 3> matrix() const = 0;

        operator Matrix<float, 3, 3>() const {
            return matrix();
        }

        operator Matrix<float, 4, 4>() const {
            Matrix<float, 4, 4> R = Matrix<float, 4, 4>::Identity();
            R.block(0, 0, 3, 3) = matrix();
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
        AngleAxis(const Matrix<float, 3, 3> &matrix) {
            float trace = matrix.trace();
            float angle = std::acos((trace - 1) / 2);
            Vector<float, 3> axis(matrix(2, 1) - matrix(1, 2), matrix(0, 2) - matrix(2, 0),
                                  matrix(1, 0) - matrix(0, 1));
            params = SafeNormalize(axis, axis.norm()) * angle;
        }

        AngleAxis(const Vector<float, 3> &angle_axis) : params(angle_axis) {

        }

        AngleAxis(float angle, const Vector<float, 3> &angle_axis) {
            params = SafeNormalize(angle_axis, angle_axis.norm()) * angle;
        }

        Matrix<float, 3, 3> exp_map(const Vector<float, 3> &r) const {
            //Rodrigues formula
            float angle = params.norm();
            if (angle < 1e-6) {
                // For very small angles, use the first-order Taylor expansion
                return Matrix<float, 3, 3>::Identity() + SkewSymmetricMatrix(r);
            }
            Matrix<float, 3, 3> K = SkewSymmetricMatrix(SafeNormalize(params, angle));
            return Matrix<float, 3, 3>::Identity() + K * std::sin(angle) + K * K * (1 - std::cos(angle));
        }

        Vector<float, 3> log_map(const Matrix<float, 3, 3> &rot) {
            float trace = rot.trace();
            float angle = std::acos((trace - 1) / 2);
            if (angle < 1e-6) {
                // For very small angles, use the first-order approximation
                return Vector<float, 3>(rot(2, 1) - rot(1, 2), rot(0, 2) - rot(2, 0), rot(1, 0) - rot(0, 1)) / 2;
            }
            Vector<float, 3> axis(rot(2, 1) - rot(1, 2), rot(0, 2) - rot(2, 0), rot(1, 0) - rot(0, 1));
            return SafeNormalize(axis, axis.norm()) * angle;
        }

        Matrix<float, 3, 3> matrix() const override {
            return exp_map(params);
        }

        Vector<float, 3> params;
    };

    struct Cayley : public Rotation {
        Cayley(const Matrix<float, 3, 3> &rot) {
            params << rot(1, 2) - rot(2, 1), rot(2, 0) - rot(0, 2), rot(0, 1) - rot(1, 0);
            params /= 2;
        }

        Cayley(const Vector<float, 3> cayley) : params(cayley) {

        }

        Matrix<float, 3, 3> matrix() const override {
            Eigen::Matrix<float, 3, 3> K = SkewSymmetricMatrix(params);
            Eigen::Matrix<float, 3, 3> Id = Eigen::Matrix<float, 3, 3>::Identity();
            return (Id - K) * (Id + K).inverse();
        }

        Vector<float, 3> params;
    };

    struct TwoAxis : public Rotation {
        TwoAxis(const Matrix<float, 3, 3> &rot) : TwoAxis(rot.col(0), rot.col(1)) {

        }

        TwoAxis(const Matrix<float, 3, 2> &two_axis) : TwoAxis(two_axis.col(0), two_axis.col(1)) {

        }

        TwoAxis(const Vector<float, 3> &axis1, const Vector<float, 3> &axis2) {
            params.col(0) = SafeNormalize(axis1, axis1.norm());
            params.col(1) = axis2 - (params.col(0).transpose() * axis2) * params.col(0);
            params.col(1) = SafeNormalize(params.col(1).eval(), params.col(1).norm());
        }

        Matrix<float, 3, 3> matrix() const override {
            Eigen::Matrix<Scalar, 3, 3> rot;
            rot.block(0, 0, 3, 2) = params;
            rot.col(2) = rot.col(0).cross(rot.col(1));
            return rot;
        }

        Matrix<Scalar, 3, 2> params;
    };
}

#endif //ENGINE24_ROTATION_H
