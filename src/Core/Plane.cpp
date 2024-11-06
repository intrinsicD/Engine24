//
// Created by alex on 18.07.24.
//

#include "Plane.h"

namespace Bcg {

    Vector<float, 3> Project(const Plane &plane, const Vector<float, 3> &point) {
        return Plane::project(plane.normal, plane.d, point);
    }

    float Distance(const Plane &plane, const Vector<float, 3> &point) {
        return Plane::signed_distance(plane.normal, plane.d, point);
    }

    float UnsignedDistance(const Plane &plane, const Vector<float, 3> &point) {
        return std::abs(Distance(plane, point));
    }

    Vector<float, 3> ClosestPoint(const Plane &plane, const Vector<float, 3> &point) {
        return Plane::closest_point(plane.normal, plane.d, point);
    }
}