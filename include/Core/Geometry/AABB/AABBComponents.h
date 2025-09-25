#pragma once
#include "AABB.h"

namespace Bcg {
    struct WorldAABB {
        AABB<float> aabb;
    };

    struct LocalAABB {
        AABB<float> aabb;
    };

    struct DirtyWorldAABB {

    };
}