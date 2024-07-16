//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_PLANE_H
#define ENGINE24_PLANE_H

#include "MatVec.h"

namespace Bcg{
    template<typename T>
    struct Plane {
        Vector<T, 3> normal;
        T d;
    };

    template<typename T>
    Vector<T, 3> closest_point(const Plane<T> &plane, const Vector<T, 3> &point) {
        return point - distance(plane, point) * plane.normal;
    }

    template<typename T>
    T distance(const Plane<T> &plane, const Vector<T, 3> &point) {
        return plane.normal.dot(point) - plane.d;
    }

    template<typename T>
    T unsigned_distance(const Plane<T> &plane, const Vector<T, 3> &point) {
        return std::abs(distance(plane, point));
    }
}

#endif //ENGINE24_PLANE_H
