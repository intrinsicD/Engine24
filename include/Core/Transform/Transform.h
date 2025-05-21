//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_TRANSFORM_H
#define ENGINE24_TRANSFORM_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    class Transform {
    public:
        struct Parameters {
            Eigen::Vector<T, 3> scale{T(1), T(1), T(1)};
            Eigen::Vector<T, 3> axis{T(0), T(0), T(1)};
            T angle{T(0)};
            Eigen::Vector<T, 3> position{T(0), T(0), T(0)};
        };

        Transform() = default;

        explicit Transform(const Parameters &params_) : params(params_) {
            params.axis.normalize();
            cached_matrix = compute_matrix(params);
        }

        const Eigen::Matrix<T, 4, 4> &matrix() const {
            if (dirty) {
                cached_matrix = compute_matrix(params);
                dirty = false;
            }
            return cached_matrix;
        }

        void set_params(const Parameters &params_) {
            params = params_;
            params.axis.normalize();
            dirty = true;
        }

        const Parameters &get_params() const {
            return params;
        }

        Parameters &get_params() {
            return params;
        }

        Eigen::Vector<T, 3> operator *(const Eigen::Vector<T, 3> &v) const {
            return (matrix() * v.homogeneous()).template head<3>();
        }

        Eigen::Matrix<T, 4, 4> operator *(const Eigen::Matrix<T, 4, 4> &m) const {
            return matrix() * m;
        }

        Transform operator *(const Transform &other) const {
            Transform result;
            result.params.scale = params.scale.cwiseProduct(other.params.scale);
            Eigen::AngleAxis<T> rot = Eigen::AngleAxis<T>(params.angle, params.axis) * Eigen::AngleAxis<T>(other.params.angle, other.params.axis);
            result.params.angle = rot.angle();
            result.params.axis = rot.axis();
            // Compose position: scale and rotate other's position, then add this position
            result.params.position = params.scale.asDiagonal() * (Eigen::AngleAxis<T>(params.angle, params.axis) * other.params.position) + params.position;
            result.dirty = true;
            return result;
        }

    private:
        static Eigen::Matrix<T, 4, 4> compute_matrix(const Parameters &p) {
            Eigen::Matrix<T, 4, 4> m = Eigen::Matrix<T, 4, 4>::Identity();
            m.block<3, 3>(0, 0) = p.scale.asDiagonal() * Eigen::AngleAxis<T>(p.angle, p.axis).toRotationMatrix();
            m.block<3, 1>(0, 3) = p.position;
            return m;
        }

        Parameters params;
        mutable Eigen::Matrix<T, 4, 4> cached_matrix = Eigen::Matrix<T, 4, 4>::Identity();
        mutable bool dirty{true};
    };
}

#endif //ENGINE24_TRANSFORM_H
