//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_COVARIANCE_H
#define ENGINE24_COVARIANCE_H

#include "MatVec.h"
#include "PropertyEigenMap.h"

namespace Bcg {
    template<typename Scalar, int N>
    struct ICovariance {
        virtual ~ICovariance() = default;

        virtual void build(const std::vector<Vector<Scalar, N>> &data, const Vector<Scalar, N> &mean) = 0;

        virtual Matrix<Scalar, N, N> inverse() const = 0;

        virtual Scalar determinant() const = 0;

        virtual Vector<Scalar, N> apply(const Vector<Scalar, N> &vec) const = 0;

        virtual Matrix<Scalar, N, N> as_matrix() const = 0;
    };

    template<typename Scalar, int N>
    struct CovarianceMatrix : public ICovariance<Scalar, N> {
        Matrix<Scalar, N, N> upper_triangular;

        CovarianceMatrix() = default;

        CovarianceMatrix(const Matrix<Scalar, N, N> &cov) : upper_triangular(
                cov.template triangularView<Eigen::Upper>()) {}

        void build(const std::vector<Vector<Scalar, N>> &data, const Vector<Scalar, N> &mean) override {
            Matrix<Scalar, N, N> cov;
            cov.setZero();
            for (const auto &vec: data) {
                cov += (vec - mean) * (vec - mean).transpose();
            }
            cov /= data.size();
            upper_triangular = cov.template triangularView<Eigen::Upper>();
        }

        Matrix<Scalar, N, N> inverse() const override {
            return as_matrix().inverse();
        }

        Scalar determinant() const override {
            return as_matrix().determinant();
        }

        Vector<Scalar, N> apply(const Vector<Scalar, N> &vec) const override {
            return as_matrix() * vec;
        }

        Matrix<Scalar, N, N> as_matrix() const override {
            Matrix<Scalar, N, N> full_matrix = upper_triangular;
            full_matrix.template triangularView<Eigen::Lower>() = upper_triangular.transpose();
            return full_matrix;
        }
    };

    template<typename Scalar, int N>
    struct CovarianceRS : public ICovariance<Scalar, N> {
        Matrix<Scalar, N, N> rotation;
        Vector<Scalar, N> scaling;

        CovarianceRS(const Matrix<Scalar, N, N> &rotation, const Vector<Scalar, N> &scaling)
                : rotation(rotation), scaling(scaling) {}

        void build(const std::vector<Vector<Scalar, N>> &data, const Vector<Scalar, N> &mean) override {
            Matrix<Scalar, N, N> cov;
            cov.setZero();
            for (const auto &vec: data) {
                cov += (vec - mean) * (vec - mean).transpose();
            }
            cov /= data.size();
            Eigen::JacobiSVD<Matrix<Scalar, N, N>> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
            rotation = svd.matrixU() * svd.matrixV().transpose();
            scaling = svd.singularValues().cwiseSqrt();
        }

        Matrix<Scalar, N, N> inverse() const override {
            Vector<Scalar, N> inv_scale = 1. / scaling;
            return rotation.transpose() * inv_scale.cwiseSquare().asDiagonal() * rotation;
        }

        Scalar determinant() const override {
            return scaling.cwiseSquare().prod();
        }

        Vector<Scalar, N> apply(const Vector<Scalar, N> &vec) const override {
            return rotation * (scaling.cwiseSquare().asDiagonal() * (rotation.transpose() * vec));
        }

        Matrix<Scalar, N, N> as_matrix() const override {
            return rotation * scaling.cwiseSquare().asDiagonal() * rotation.transpose();
        }
    };

    template<typename Scalar>
    struct CovarianceRS3 : public ICovariance<Scalar, 3> {
        Vector<Scalar, 3> scaling;
        Eigen::AngleAxis<Scalar> angle_axis;

        CovarianceRS3(const Vector<Scalar, 3> &scaling, const Eigen::AngleAxis<Scalar> &angle_axis)
                : scaling(scaling), angle_axis(angle_axis) {}

        void build(const std::vector<Vector<Scalar, 3>> &data, const Vector<Scalar, 3> &mean) override {
            Matrix<Scalar, 3, 3> cov;
            cov.setZero();
            for (const auto &vec: data) {
                cov += (vec - mean) * (vec - mean).transpose();
            }
            cov /= data.size();
            Eigen::JacobiSVD<Matrix<Scalar, 3, 3>> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
            scaling = svd.singularValues().cwiseSqrt();
            angle_axis = Eigen::AngleAxis<Scalar>(svd.matrixU() * svd.matrixV().transpose());
        }

        Matrix<Scalar, 3, 3> inverse() const override {
            Vector<Scalar, 3> inv_scale = 1. / scaling;
            Matrix<Scalar, 3, 3> rot = angle_axis.toRotationMatrix();
            return rot.transpose() * inv_scale.cwiseSquare().asDiagonal() * rot;
        }

        Scalar determinant() const override {
            return scaling.cwiseSquare().prod();
        }

        Vector<Scalar, 3> apply(const Vector<Scalar, 3> &vec) const override {
            Matrix<Scalar, 3, 3> rot = angle_axis.toRotationMatrix();
            return rot * (scaling.cwiseSquare().asDiagonal() * (rot.transpose() * vec));
        }

        Matrix<Scalar, 3, 3> as_matrix() const override {
            Matrix<Scalar, 3, 3> rot = angle_axis.toRotationMatrix();
            return rot * scaling.cwiseSquare().asDiagonal() * rot.transpose();
        }
    };
}

#endif //ENGINE24_COVARIANCE_H
