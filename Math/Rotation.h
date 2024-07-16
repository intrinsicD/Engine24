//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_ROTATION_H
#define ENGINE24_ROTATION_H

#include "MatVec.h"

namespace Bcg {
    using Quaternion = Eigen::Quaternion<float>;

    template<typename Scalar>
    Matrix<Scalar, 3, 3> CrossProductMatrix(const Vector<Scalar, 3> &vector) {
        Matrix<Scalar, 3, 3> matrix;
        matrix << 0, -vector[2], vector[1],
                vector[2], 0, -vector[0],
                -vector[1], vector[0], 0;
        return matrix;
    }

    template<class Derived>
    struct Rotation {
        using Scalar = typename Derived::Scalar;

        Matrix<Scalar, 3, 3> to_matrix() const {
            return derived.to_matrix();
        }

        void from_matrix(const Matrix<Scalar, 3, 3> &matrix) {
            derived.from_matrix(matrix);
        }

        Matrix<Scalar, 3, 3> &operator*(const Matrix<Scalar, 3, 3> &other) {
            return derived.to_matrix() * other;
        }

        Vector<Scalar, 3> &operator*(const Vector<Scalar, 3> &other) {
            return derived.to_matrix() * other;
        }

        Scalar *data() {
            return derived.data();
        }

        const Scalar *data() const {
            return derived.data();
        }

        Derived derived;
    };

    template<typename Scalar_>
    struct AngleAxis {
        using Scalar = Scalar_;
        Matrix<Scalar, 3, 3> to_matrix() const {
            //Rodrigues formula
            Scalar angle = params.norm();
            Eigen::Matrix<Scalar, 3, 3> K = CrossProductMatrix(SafeNormalize(params, angle));
            return Eigen::Matrix<Scalar, 3, 3>::Identity() + K * std::sin(angle) + K * K * (1 - std::cos(angle));
        }

        void from_matrix(const Matrix<Scalar, 3, 3> &matrix) {
            Scalar trace = matrix.trace();
            Scalar angle = std::acos((trace - 1) / 2);
            Eigen::Vector<Scalar, 3> axis(matrix(2, 1) - matrix(1, 2), matrix(0, 2) - matrix(2, 0),
                                     matrix(1, 0) - matrix(0, 1));
            params = SafeNormalize(axis, axis.norm()) * angle;
        }

        Vector<Scalar, 3> params;
    };

    template<typename Scalar_>
    struct Cayley {
        using Scalar = Scalar_;
        Matrix<Scalar, 3, 3> to_matrix() const {
            Eigen::Matrix<Scalar, 3, 3> K = CrossProductMatrix(params);
            Eigen::Matrix<Scalar, 3, 3> Id = Eigen::Matrix<Scalar, 3, 3>::Identity();
            return (Id - K) * (Id + K).inverse();
        }

        void from_matrix(const Matrix<Scalar, 3, 3> &matrix) {
            params << matrix(1, 2) - matrix(2, 1), matrix(2, 0) - matrix(0, 2), matrix(0, 1) - matrix(1, 0);
            params /= 2;
        }

        Vector<Scalar, 3> params;
    };

    template<typename Scalar_>
    struct TwoAxis {
        using Scalar = Scalar_;
        Matrix<Scalar, 3, 3> to_matrix() const {
            Eigen::Matrix<Scalar, 3, 3> rot;
            rot.col(0) = SafeNormalize(params.col(0), params.col(0).norm());
            rot.col(1) = params.col(1) - (rot.col(0).transpose() * params.col(1)) * rot.col(0);
            rot.col(1) = SafeNormalize(rot.col(1), rot.col(1).norm());
            rot.col(2) = rot.col(0).cross(rot.col(1));
            return rot;
        }

        void from_matrix(const Matrix<Scalar, 3, 3> &matrix) {
            params = matrix.leftCols(2);
        }

        Matrix<Scalar, 3, 2> params;
    };
}

#endif //ENGINE24_ROTATION_H
