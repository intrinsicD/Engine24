//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_TRANSFORMUTILS_H
#define ENGINE24_TRANSFORMUTILS_H

#include "Transform.h"
#include "StringTraits.h"
#include "glm/gtx/string_cast.hpp"

namespace Bcg{

    template<>
    struct StringTraits<Transform> {
        static std::string ToString(const Transform &t) {
            std::stringstream ss;
            ss << "Transform: \n";
            ss << "Local Matrix: \n" << glm::to_string(t.local()) << "\n";
            ss << "World Matrix: \n" << glm::to_string(t.world()) << "\n";
            return ss.str();
        }
    };

    void pre_transform(Transform &t, glm::mat4 &other);

    void post_transform(Transform &t, glm::mat4 &other);

    TransformParameters decompose(const glm::mat4 &matrix);

    glm::mat4 compose(const TransformParameters &params);

    void set_transform_params(Transform &t, const TransformParameters &params);
}

#endif //ENGINE24_TRANSFORMUTILS_H
