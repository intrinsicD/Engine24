//
// Created by alex on 27.07.24.
//

#ifndef ENGINE24_AABBPOOL_H
#define ENGINE24_AABBPOOL_H

#include "AABB.h"
#include "ResourcePool.h"

namespace Bcg {
    using AABBPool = ResourcePool<AABB>;
    using AABBHandle = ResourceHandle<AABB, AABBPool>;
}

#endif //ENGINE24_AABBPOOL_H
