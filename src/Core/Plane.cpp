//
// Created by alex on 18.07.24.
//

#include "Plane.h"

namespace Bcg{
    float Distance(const Plane &plane, const Vector<float, 3> &point) {
        return glm::dot(plane.normal, point) - plane.d;
    }

    float UnsignedDistance(const Plane &plane, const Vector<float, 3> &point) {
        return std::abs(Distance(plane, point));
    }

    Vector<float, 3> ClosestPoint(const Plane &plane, const Vector<float, 3> &point) {
        return point - Distance(plane, point) * plane.normal;
    }
}