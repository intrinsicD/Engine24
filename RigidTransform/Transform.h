//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_TRANSFORM_H
#define ENGINE24_TRANSFORM_H

#include "RigidTransform.h"

namespace Bcg {
    struct Transform {
        RigidTransform local;
        RigidTransform world;
        bool dirty = true;

        Transform() : local(RigidTransform::Identity()), world(RigidTransform::Identity()) {

        }

        void set_local_identity() {
            set_local(Matrix<float, 4, 4>::Identity());
        }

        void set_local(const Matrix<float, 4, 4> &t) {
            local.matrix() = t;
            dirty = true;
        }

        [[nodiscard]] const Matrix<float, 4, 4> &matrix() const {
            return world.matrix();
        }

        void update(const Matrix<float, 4, 4> &parent_world) {
            world.matrix() = parent_world * local.matrix();
            dirty = true;
        }

        [[nodiscard]] const float *data() const {
            return world.data();
        }
    };
}

#endif //ENGINE24_TRANSFORM_H
