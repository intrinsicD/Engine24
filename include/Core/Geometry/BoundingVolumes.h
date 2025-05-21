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
        PoolHandle<AABB> h_aabb;
        PoolHandle<Sphere> h_sphere;
    };
}
#endif //ENGINE24_BOUNDINGVOLUMES_H
