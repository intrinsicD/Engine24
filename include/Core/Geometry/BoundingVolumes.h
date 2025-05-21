//
// Created by alex on 12.11.24.
//

#ifndef ENGINE24_BOUNDINGVOLUMES_H
#define ENGINE24_BOUNDINGVOLUMES_H

#include "Pool.h"
#include "PoolHandle.h"
#include "AABB.h"
#include "Sphere.h"

namespace Bcg{
    // Component container for handles to different bounding volumes
    struct BoundingVolumes{
        PoolHandle<AABB<float, 3>> h_aabb;
        PoolHandle<Sphere<float, 3>> h_sphere;
    };
}
#endif //ENGINE24_BOUNDINGVOLUMES_H
