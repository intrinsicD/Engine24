#pragma once

#include "eigen3/Eigen/Dense"

namespace Bcg {
    template<typename T>
    struct CovarianceInterface {
        Eigen::Vector<T, 3> &scale;
        Eigen::Quaternion<T> &quaternion;

        CovarianceInterface(Eigen::Vector<T, 3> &scale, Eigen::Quaternion<T> &quaternion) : scale(scale),
            quaternion(quaternion) {
        }

        Eigen::Matrix<T, 3, 3> get_covariance_matrix() const {
            Eigen::Matrix<T, 3, 3> rotation_matrix = quaternion.toRotationMatrix();
            Eigen::Matrix<T, 3, 3> scale_matrix = Eigen::Matrix<T, 3, 3>::Zero();
            scale_matrix(0, 0) = scale[0] * scale[0];
            scale_matrix(1, 1) = scale[1] * scale[1];
            scale_matrix(2, 2) = scale[2] * scale[2];
            return rotation_matrix * scale_matrix * rotation_matrix.transpose();
        }

        Eigen::Matrix<T, 3, 3> get_inverse_covariance_matrix() const {
            Eigen::Matrix<T, 3, 3> rotation_matrix = quaternion.toRotationMatrix();
            Eigen::Matrix<T, 3, 3> inv_scale_sq = Eigen::Matrix<T, 3, 3>::Zero();
            // Avoid division by zero for scales that are zero
            inv_scale_sq(0, 0) = (scale[0] != 0) ? T(1) / (scale[0] * scale[0]) : T(0);
            inv_scale_sq(1, 1) = (scale[1] != 0) ? T(1) / (scale[1] * scale[1]) : T(0);
            inv_scale_sq(2, 2) = (scale[2] != 0) ? T(1) / (scale[2] * scale[2]) : T(0);
            return rotation_matrix * inv_scale_sq * rotation_matrix.transpose();
        }

        void set_from_covariance_matrix(const Eigen::Matrix<T, 3, 3> &covariance) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3> > eigen_solver(covariance);

            Eigen::Matrix<T, 3, 3> eigen_vectors = eigen_solver.eigenvectors();
            Eigen::Vector<T, 3> eigen_values = eigen_solver.eigenvalues();

            if (eigen_vectors.determinant() < 0) {
                // Flip one of the axes to enforce a right-handed coordinate system.
                eigen_vectors.col(0) *= -1;
            }

            scale[0] = std::sqrt(std::max(eigen_values[0], T(0)));
            scale[1] = std::sqrt(std::max(eigen_values[1], T(0)));
            scale[2] = std::sqrt(std::max(eigen_values[2], T(0)));
            quaternion = Eigen::Quaternion<T>(eigen_vectors);
        }

        T determinant() const {
            T s = scale[0] * scale[1] * scale[2];
            return s * s;
        }

        Eigen::Vector<T, 3> operator*(const Eigen::Vector<T, 3>& v) const {
            Eigen::Vector<T, 3> v_local = quaternion.inverse() * v;
            v_local = v_local.array() * scale.array().square();
            return quaternion * v_local;
        }
    };

    template<typename T>
    struct CovarianceData {
        CovarianceData(const Eigen::Vector<T, 3> &scale = Eigen::Vector<T, 3>::Ones(),
                       const Eigen::Quaternion<T> &quaternion = Eigen::Quaternion<T>::Identity())
            : scale(scale), quaternion(quaternion) {
        }

        Eigen::Vector<T, 3> scale;
        Eigen::Quaternion<T> quaternion;
    };

    template<typename T>
    struct CovarianceMatrix {
        CovarianceMatrix() : data(), interface(data.scale, data.quaternion) {
        }

        CovarianceMatrix(const CovarianceData<T> &initial_data) : data(initial_data),
                                                                  interface(data.scale, data.quaternion) {
        }

        CovarianceMatrix(const Eigen::Vector<T, 3> &scale,
                         const Eigen::Quaternion<T> &quaternion) : data(scale, quaternion),
                                                                   interface(data.scale, data.quaternion) {
        }

        CovarianceMatrix(const Eigen::Matrix<T, 3, 3> &covariance_matrix) : data(), interface(data.scale, data.quaternion) {
            interface.set_from_covariance_matrix(covariance_matrix);
        }

        Eigen::Vector<T, 3> operator*(const Eigen::Vector<T, 3>& v) const {
            return interface * v;
        }

        CovarianceData<T> data;
        CovarianceInterface<T> interface;
    };
}
