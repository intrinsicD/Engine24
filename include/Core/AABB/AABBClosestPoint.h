//
// Created by alex on 10.09.24.
//

#ifndef ENGINE24_AABBCLOSESTPOINT_H
#define ENGINE24_AABBCLOSESTPOINT_H

#include "AABBStruct.h"

namespace Bcg {
    template<typename T, int N>
    inline Vector<T, N> AABBClosestPoint(const Vector<T, N> &min, const Vector<T, N> &max, const Vector<T, N> &point) {
        return point.cwiseMax(min).cwiseMin(max);
    }

    template<typename T, int N>
    inline Vector<T, N> ClosestPoint(const AABBStruct<T, N> &aabb, const Vector<T, N> &point) {
        return AABBClosestPoint(aabb.min, aabb.max, point);
    }
}

#endif //ENGINE24_AABBCLOSESTPOINT_H
