//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_CAMERA_H
#define ENGINE24_CAMERA_H

#include "MatVec.h"

namespace Bcg {
    struct Camera {
        Matrix<float, 4, 4> view;
        Matrix<float, 4, 4> proj;
        bool dirty = false;
    };
}

#endif //ENGINE24_CAMERA_H
