//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_FRUSTUM_H
#define ENGINE24_FRUSTUM_H

#include "Plane.h"

namespace Bcg {
    template<typename T>
    struct Frustum {
        union {
            Matrix<T, 6, 4> planes;
            Plane<T> plane[6];
        };
    };

    template<typename T>
    Vector<T, 3> closest_point(const Frustum<T> &frustum, const Vector<T, 3> &point) {
        //TODO this is harder than i first thought
        return {};
    }

    template<typename T>
    T distance(const Frustum<T> &frustum, const Vector<T, 3> &point) {
        //if point is inside, the distance should be negative
        //if point is outside, the distance should be positive
        //TODO this function is probably wrong. There are more conditionals depending on each plane...
        T min_distance = std::numeric_limits<T>::max();

        for (int i = 0; i < 6; ++i) {
            T d = distance(frustum.plane[i], point);
            if (std::abs(d) < std::abs(min_distance)) {
                min_distance = d;
            }
        }

        return min_distance;
    }

    template<typename T>
    static T unsigned_distance(const Frustum<T> &frustum, const Vector<T, 3> &point) {
        return std::abs(distance(frustum, point));
    }
}

#endif //ENGINE24_FRUSTUM_H
