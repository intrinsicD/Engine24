//
// Created by alex on 5/21/25.
//

#ifndef TRANSFORMUTILS_H
#define TRANSFORMUTILS_H

#include "Transform.h"
#include "Command.h"

namespace Bcg::Transform {
    // Decomposes a 4x4 affine transform matrix into position, scale, and rotation (angle/axis).
    // Assumes no shear or perspective.
    template<typename T>
    Eigen::Matrix<T, 4, 4> GetTransform(const Parameters<T> &p) {
        Eigen::Matrix<T, 4, 4> m = Eigen::Matrix<T, 4, 4>::Identity();
        m.template block<3, 3>(0, 0) = p.scale.asDiagonal() * Eigen::AngleAxis<T>(p.angle, p.axis).toRotationMatrix();
        m.template block<3, 1>(0, 3) = p.position;
        return m;
    }

    template<typename T>
    Parameters<T> GetParams(const Eigen::Matrix<T, 4, 4> &matrix) {
        Parameters<T> params;
        //use svd to extract scale and rotation
        Eigen::JacobiSVD<Eigen::Matrix<T, 3, 3>> svd(matrix.template block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        params.scale = svd.singularValues();
        Eigen::Matrix<T, 3, 3> rotation = svd.matrixU() * svd.matrixV().transpose();
        Eigen::AngleAxis<T> angle_axis(rotation);
        params.angle = angle_axis.angle();
        params.axis = angle_axis.axis().normalized();
        params.position = matrix.template block<3, 1>(0, 3);
        return params;
    }

    template<typename T>
    Parameters<T> Inverse(const Parameters<T> &params) {
        Parameters<T> inv_params;
        inv_params.scale = T(1) / params.scale.array();
        Eigen::AngleAxis<T> rot(-params.angle, params.axis);
        inv_params.angle = rot.angle();
        inv_params.axis = rot.axis();
        // Inverse position: scale and rotate this position, then negate
        inv_params.position = -inv_params.scale.asDiagonal() * (rot * params.position);
        return inv_params;
    }

    void UpdateCachedLocal();
    void UpdateCachedWorld();
    void Update();
}

namespace Bcg::Commands {
    struct SetParameters : public AbstractCommand {
        SetParameters(entt::entity entity_id, const Transform::Parameters<float> &params)
                : AbstractCommand("SetParameters"), entity_id(entity_id), params(params) {

        }

        void execute() const override;

        entt::entity entity_id;
        Transform::Parameters<float> params;
    };

    struct UpdateCachedWorld : public AbstractCommand {
        explicit UpdateCachedWorld(entt::entity entity_id) : AbstractCommand("UpdateCachedWorld"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };
}
#endif //TRANSFORMUTILS_H
