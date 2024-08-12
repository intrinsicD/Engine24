//
// Created by alex on 18.07.24.
//

#include "OBB.h"

namespace Bcg{
    Vector<float, 3> ClosestPoint(const OBB &obb, const Vector<float, 3> &point) {
        Vector<float, 3> local_point = obb.orientation.matrix().inverse() * (point - obb.center);
        Vector<float, 3> closest = AABB::closest_point(-obb.half_extent, obb.half_extent, local_point);
        return obb.orientation.matrix() * closest + obb.center;
    }

    float Distance(const OBB &obb, const Vector<float, 3> &point) {
        Vector<float, 3> local_point = obb.orientation.matrix().inverse() * (point - obb.center);
        Vector<float, 3> closest_point;
        for (int i = 0; i < 3; ++i) {
            float dist = local_point[i];
            dist = std::max(dist, -obb.half_extent[i]);
            dist = std::min(dist, obb.half_extent[i]);
            closest_point[i] = dist;
        }
        return (local_point - closest_point).norm();
    }

    float UnsignedDistance(const OBB &obb, const Vector<float, 3> &point) {
        return std::abs(Distance(obb, point));
    }
}