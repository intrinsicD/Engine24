//
// Created by alex on 02.06.25.
//

#include "TransformUtils.h"
#include <glm/gtx/matrix_decompose.hpp>

namespace Bcg {
    void pre_transform(TransformComponent &t, glm::mat4 &other) {
        glm::mat4 result = t.matrix() * other;
        t = decompose(result);
    }

    void post_transform(TransformComponent &t, glm::mat4 &other) {
        glm::mat4 result = other * t.matrix();
        t = decompose(result);
    }

    TransformComponent decompose(const glm::mat4 &matrix) {
        glm::vec3 scale, skew;
        glm::vec4 perspective;
        glm::quat orientation;
        glm::vec3 translation;
        glm::decompose(matrix, scale, orientation, translation, skew, perspective);
        glm::vec3 angle_axis = glm::eulerAngles(orientation);
        return {scale, angle_axis, translation};
    }

    glm::mat4 compose(const TransformComponent &transform) {
        glm::mat4 trans = glm::translate(glm::mat4(1.0f), transform.position);
        glm::mat4 rot = glm::mat4_cast(transform.rotation);
        glm::mat4 sc = glm::scale(glm::mat4(1.0f), transform.scale);
        return trans * rot * sc;
    }

    void ScaleAndCenterAt(TransformComponent &t, const glm::vec3 &center_world, float scale) {
        t.position = center_world;
        t.scale = glm::vec3(scale);
    }
}