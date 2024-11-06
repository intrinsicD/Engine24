//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_PLANE_H
#define ENGINE24_PLANE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    struct PlaneBase {
        Vector<T, 3> normal;
        T d;

        inline static float signed_distance(const Vector<T, 3> &normal, T d, const Vector<T, 3> &point) {
            return glm::dot(normal, point) + d;
        }

        inline static Vector<T, 3> project(const Vector<T, 3> &normal, T d, const Vector<T, 3> &point) {
            return point - signed_distance(normal, d, point) * normal;
        }

        inline static Vector<T, 3> closest_point(const Vector<T, 3> &normal, T d, const Vector<T, 3> &point) {
            return point - signed_distance(normal, d, point) * normal;
        }
    };

    using Planef = PlaneBase<float>;
    using Plane = Planef;

    Vector<float, 3> Project(const Plane &plane, const Vector<float, 3> &point);

    float Distance(const Plane &plane, const Vector<float, 3> &point);

    float UnsignedDistance(const Plane &plane, const Vector<float, 3> &point);

    Vector<float, 3> ClosestPoint(const Plane &plane, const Vector<float, 3> &point);
}

#endif //ENGINE24_PLANE_H
