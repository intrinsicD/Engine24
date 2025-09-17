//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_TRANSFORMUTILS_H
#define ENGINE24_TRANSFORMUTILS_H

#include "TransformComponent.h"
#include "StringTraits.h"
#include "glm/gtx/string_cast.hpp"

namespace Bcg{

    template<>
    struct StringTraits<TransformComponent> {
        static std::string ToString(const TransformComponent &t) {
            std::stringstream ss;
            ss << "Transform: \n";
            ss << "Position: " << glm::to_string(t.position) << "\n";
            ss << "Rotation: " << glm::to_string(t.rotation) << "\n";
            ss << "Scale: " << glm::to_string(t.scale) << "\n";
            return ss.str();
        }
    };

    void pre_transform(TransformComponent &t, glm::mat4 &other);

    void post_transform(TransformComponent &t, glm::mat4 &other);

    TransformComponent decompose(const glm::mat4 &matrix);

    glm::mat4 compose(const TransformComponent &component);

    void ScaleAndCenterAt(TransformComponent &t, const glm::vec3 &center, float scale);
}

#endif //ENGINE24_TRANSFORMUTILS_H
