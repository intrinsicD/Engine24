//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_OBB_H
#define ENGINE24_OBB_H

#include "../Rotation.h"
#include "AABB.h"

namespace Bcg {
    template<typename T>
    struct OBB {
        Vector<T, 3> center;
        Rotation<AngleAxis<T>> orientation;
        Vector<T, 3> half_extent;
    };

    template<typename T>
    Vector<T, 3> closest_point(const OBB<T> &obb, const Vector<T, 3> &point) {
        Vector<T, 3> local_point = obb.orientation.inverse() * (point - obb.center);
        Vector<T, 3> closest = AABB<T>::closest_point(-obb.half_extent, obb.half_extent, local_point);
        return obb.orientation * closest + obb.center;
    }

    template<typename T>
    T distance(const OBB<T> &obb, const Vector<T, 3> &point) {
        Vector<T, 3> local_point = obb.orientation.inverse() * (point - obb.center);
        Vector<T, 3> closest_point;
        for (int i = 0; i < 3; ++i) {
            T dist = local_point[i];
            dist = std::max(dist, -obb.half_extent[i]);
            dist = std::min(dist, obb.half_extent[i]);
            closest_point[i] = dist;
        }
        return (local_point - closest_point).norm();
    }

    template<typename T>
    T unsigned_distance(const OBB<T> &obb, const Vector<T, 3> &point) {
        return std::abs(OBB<T>::distance(obb, point));
    }
}

#endif //ENGINE24_OBB_H
