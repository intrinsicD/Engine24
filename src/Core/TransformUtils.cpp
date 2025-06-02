//
// Created by alex on 02.06.25.
//

#include "TransformUtils.h"

namespace Bcg{
    void pre_transform(Transform &t, glm::mat4 &other) {
        t.set_local(t.local() * other);
    }

    void post_transform(Transform &t, glm::mat4 &other) {
        t.set_local(other * t.local());
    }

    TransformParameters decompose(const glm::mat4 &matrix) {
        glm::vec3 scale, skew;
        glm::vec4 perspective;
        glm::quat orientation;
        glm::vec3 translation;
        glm::decompose(matrix, scale, orientation, translation, skew, perspective);
        glm::vec3 angle_axis = glm::eulerAngles(orientation);
        return {scale, angle_axis, translation};
    }

    glm::mat4 compose(const TransformParameters &params) {
        glm::mat4 matrix(1.0f);
        matrix = glm::translate(matrix, params.position);
        matrix = glm::rotate(matrix, params.angle_axis.x, glm::vec3(1.0f, 0.0f, 0.0f));
        matrix = glm::rotate(matrix, params.angle_axis.y, glm::vec3(0.0f, 1.0f, 0.0f));
        matrix = glm::rotate(matrix, params.angle_axis.z, glm::vec3(0.0f, 0.0f, 1.0f));
        matrix = glm::scale(matrix, params.scale);
        return matrix;
    }

    void set_transform_params(Transform &t, const TransformParameters &params){
        t.set_local(compose(params));
    }
}