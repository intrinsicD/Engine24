//
// Created by alex on 25.11.24.
//

#ifndef AABBPOOL_H
#define AABBPOOL_H

#include "Pool.h"
#include "AABB.h"

namespace Bcg{
    using AABBPool = Pool<AABB>;
    using AABBHandle = PoolHandle<AABB>;
}

#endif //AABBPOOL_H
