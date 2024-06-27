//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_PRIMITIVES_H
#define ENGINE24_PRIMITIVES_H

#include "MatVec.h"

namespace Bcg {
    struct Frustum {
        Matrix<float, 6, 4> planes;
    };

    struct Sphere {
        Vector<float, 3> center;
        float radius;
    };

    struct AABB {
        Vector<float, 3> min, max;
    };

    struct OBB {

    };

    struct ConvexHull {

    };
}

#endif //ENGINE24_PRIMITIVES_H
