//
// Created by alex on 10.09.24.
//

#ifndef ENGINE24_AABBDISTANCEPOINT_H
#define ENGINE24_AABBDISTANCEPOINT_H

#include "AABBClosestPoint.h"

namespace Bcg {
    template<typename T, int N>
    inline Vector<T, N> Diff(const AABBbase<T> &aabb, const Vector<T, N> &point) {
        return ClosestPoint(aabb, point) - point;
    }

    template<typename T, int N>
    inline T L2Distance(const AABBbase<T> &aabb, const Vector<T, N> &point) {
        return Diff(aabb, point).norm();
    }

    template<typename T, int N>
    inline T L1Distance(const AABBbase<T> &aabb, const Vector<T, N> &point) {
        return Diff(aabb, point).cwiseAbs().sum();
    }

    template<typename T, int N>
    inline T LInfDistance(const AABBbase<T> &aabb, const Vector<T, N> &point) {
        return Diff(aabb, point).cwiseAbs().maxCoeff();
    }

    template<typename T, int N>
    inline T SquaredL2Distance(const AABBbase<T> &aabb, const Vector<T, N> &point) {
        return Diff(aabb, point).squaredNorm();
    }
}

#endif //ENGINE24_AABBDISTANCEPOINT_H
