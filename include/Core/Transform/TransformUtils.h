//
// Created by alex on 5/21/25.
//

#ifndef TRANSFORMUTILS_H
#define TRANSFORMUTILS_H

#include "Transform.h"
#include "Engine.h"
#include "Logger.h"

namespace Bcg {
    // Decomposes a 4x4 affine transform matrix into position, scale, and rotation (angle/axis).
    // Assumes no shear or perspective.
    template<typename T>
    typename Transform<T>::Parameters Decompose(const Eigen::Matrix<T, 4, 4> &matrix) {
        typename Transform<T>::Parameters params;
        params.position = matrix.template block<3, 1>(0, 3);
        Eigen::Matrix<T, 3, 3> m = matrix.template block<3, 3>(0, 0);

        // Extract scale
        params.scale = m.colwise().norm();

        // Avoid division by zero
        Eigen::Matrix<T, 3, 3> rotation_matrix = m;
        for (int i = 0; i < 3; ++i) {
            if (params.scale[i] > T(0)) {
                rotation_matrix.col(i) /= params.scale[i];
            } else {
                rotation_matrix.col(i).setZero();
            }
        }

        Eigen::AngleAxis<T> angle_axis(rotation_matrix);
        params.angle = angle_axis.angle();
        params.axis = angle_axis.axis();
        return params;
    }

    template<typename T>
    void SetParameters(entt::entity entity_id, const typename Transform<T>::Parameters &params) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Invalid entity ID: {}", entity_id);
            return;
        }
        if (!Engine::has<Transform<T>>(entity_id)) {
            Log::Warn("Entity does not have a Transform component: {}", entity_id);
            return;
        }

        auto &transform = Engine::State().get<Transform<T>>(entity_id);
        transform.set_params(params);

        MarkComponentDirty<Transform<float>>(entity_id);
    }
}
#endif //TRANSFORMUTILS_H
