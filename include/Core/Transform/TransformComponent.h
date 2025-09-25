//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_TRANSFORMCOMPONENT_H
#define ENGINE24_TRANSFORMCOMPONENT_H

#include "MatVec.h"

namespace Bcg{
    struct TransformComponent {
        glm::vec3 position = { 0.0f, 0.0f, 0.0f };
        glm::quat rotation = { 1.0f, 0.0f, 0.0f, 0.0f }; // W,X,Y,Z identity quaternion
        glm::vec3 scale    = { 1.0f, 1.0f, 1.0f };

        glm::mat4 matrix() const {
            // The order of operations is crucial: Scale -> Rotate -> Translate
            glm::mat4 trans = glm::translate(glm::mat4(1.0f), position);
            glm::mat4 rot = glm::mat4_cast(rotation);
            glm::mat4 sc = glm::scale(glm::mat4(1.0f), scale);
            return trans * rot * sc;
        }
    };

    struct DirtyLocalTransform{};
}

#endif //ENGINE24_TRANSFORMCOMPONENT_H
