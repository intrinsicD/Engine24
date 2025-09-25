//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_WORLDTRANSFORMCOMPONENT_H
#define ENGINE24_WORLDTRANSFORMCOMPONENT_H

#include "../MatVec.h"

namespace Bcg{
    struct WorldTransformComponent {
       Matrix<float, 4, 4> world_transform = Matrix<float, 4, 4>(1.0f); // Identity matrix by default
    };

    struct DirtyWorldTransform{};
}

#endif //ENGINE24_WORLDTRANSFORMCOMPONENT_H
