//
// Created by alex on 25.11.24.
//

#ifndef AABBPOOL_H
#define AABBPOOL_H

#include "Pool.h"
#include "AABB.h"

namespace Bcg{
    using AABBPool = Pool<AABB<float, 3>>;
    using AABBHandle = PoolHandle<AABB<float, 3>>;
}

#endif //AABBPOOL_H
