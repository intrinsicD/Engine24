//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_PLANEUTILS_H
#define ENGINE24_PLANEUTILS_H

#include "Plane.h"

namespace Bcg{
    template<typename T, int N>
    Eigen::Vector<T, N> ClosestPoint(const Plane<T, N> &plane, const Eigen::Vector<T, N> &point) {
        return point - (plane.normal.dot(point) + plane.d) / plane.normal.squaredNorm() * plane.normal;
    }

    template<typename T, int N>
    T Distance(const Plane<T, N> &plane, const Eigen::Vector<T, N> &point) {
        return (point - ClosestPoint(plane, point)).norm();
    }
}

#endif //ENGINE24_PLANEUTILS_H
