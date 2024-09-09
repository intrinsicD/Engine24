//
// Created by alex on 10.09.24.
//

#ifndef ENGINE24_AABBCLOSESTPOINT_H
#define ENGINE24_AABBCLOSESTPOINT_H

#include "AABB.h"

namespace Bcg{
    template<typename T, int N>
    Vector<T, N> ClosestPoint(const AABBbase<T> &aabb, const Vector<T, N> &point) {
        Vector<T, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = std::clamp(point[i], aabb.min[i], aabb.max[i]);
        }
        return result;
    }

    template<typename T>
    inline Vector<T, 3> ClosestPoint(const AABBbase<T> &aabb, const Vector<T, 3> &point) {
        return point.cwiseMax(aabb.min).cwiseMin(aabb.max);
    }
}

#endif //ENGINE24_AABBCLOSESTPOINT_H
