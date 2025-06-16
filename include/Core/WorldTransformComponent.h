//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_WORLDTRANSFORMCOMPONENT_H
#define ENGINE24_WORLDTRANSFORMCOMPONENT_H

#include "MatVec.h"

namespace Bcg{
    struct WorldTransformComponent {
        glm::mat4 world_transform = glm::mat4(1.0f); // Identity matrix by default
    };
}

#endif //ENGINE24_WORLDTRANSFORMCOMPONENT_H
