//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_TRANSFORM_H
#define ENGINE24_TRANSFORM_H

#include "RigidTransform.h"

namespace Bcg {
    struct Transform {
        RigidTransform local;

        Transform() : local(RigidTransform::Identity()), m_world(RigidTransform::Identity()), dirty(false) {

        }

        void set_local_identity() {
            set_local(Matrix<float, 4, 4>::Identity());
        }

        bool is_dirty() const {
            return dirty;
        }

        void mark_dirty() {
            dirty = true;
        }

        void mark_clean() {
            dirty = false;
        }

        void set_local(const Matrix<float, 4, 4> &t) {
            local.matrix() = t;
            mark_dirty();
        }

        void set_local(const RigidTransform &t) {
            local = t;
            mark_dirty();
        }

        [[nodiscard]] const RigidTransform &world() const {
            return m_world;
        }

        void update_world(const Matrix<float, 4, 4> &parent_world) {
            m_world.matrix() = parent_world * local.matrix();
        }

        void update_world(const RigidTransform &parent_world) {
            m_world = parent_world * local;
        }

        [[nodiscard]] const float *data() const {
            return m_world.data();
        }

    private:
        bool dirty;
        RigidTransform m_world;
    };
}

#endif //ENGINE24_TRANSFORM_H
