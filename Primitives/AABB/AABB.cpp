//
// Created by alex on 18.07.24.
//

#include "AABB.h"

namespace Bcg {
    AABB &Grow(AABB &aabb, const Vector<float, 3> &point) {
        aabb.min = aabb.min.cwiseMin(point);
        aabb.max = aabb.max.cwiseMax(point);
        return aabb;
    }

    AABB &Build(AABB &aabb, const std::vector<Vector<float, 3>> &points) {
        aabb = AABB();
        for (auto &point: points) {
            Grow(aabb, point);
        }
        return aabb;
    }

    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point) {
        return AABB::closest_point(aabb.min, aabb.max, point);
    }

    float Distance(const AABB &aabb, const Vector<float, 3> &point) {
        return (point - ClosestPoint(aabb, point)).norm();
    }

    float UnsignedDistance(const AABB &aabb, const Vector<float, 3> &point) {
        return std::abs(Distance(aabb, point));
    }
}