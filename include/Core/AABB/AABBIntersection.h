//
// Created by alex on 13.09.24.
//

#ifndef ENGINE24_AABBINTERSECTION_H
#define ENGINE24_AABBINTERSECTION_H

#include "AABBStruct.h"

namespace Bcg {
    template<typename Scalar, int N>
    bool Intersects(const AABBStruct<Scalar, N> &aabb1, const AABBStruct<Scalar, N> &aabb2) {
        for (int i = 0; i < N; ++i) {
            if (aabb1.max[i] < aabb2.min[i] || aabb1.min[i] > aabb2.max[i]) {
                return false; // No overlap on this axis
            }
        }
        return true; // Overlaps on all axes
    }
}

#endif //ENGINE24_AABBINTERSECTION_H
