//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_TRANSFORM_H
#define ENGINE24_TRANSFORM_H

#include "RigidTransform.h"
#include "glm/gtc/type_ptr.hpp"

namespace Bcg {
    struct Transform {
        glm::mat4 local;

        Transform() : local(glm::mat4(1.0f)), m_world(glm::mat4(1.0f)), dirty(false) {

        }

        void set_local_identity() {
            set_local(glm::mat4(1.0f));
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
            local = t;
            mark_dirty();
        }

        [[nodiscard]] const glm::mat4 &world() const {
            return m_world;
        }

        void update_world(const glm::mat4 &parent_world) {
            m_world = parent_world * local;
        }

        [[nodiscard]] const float *data() const {
            return glm::value_ptr(m_world);
        }

    private:
        bool dirty;
        glm::mat4 m_world;
    };
}

#endif //ENGINE24_TRANSFORM_H
