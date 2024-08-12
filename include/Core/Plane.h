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
    };

    using Planef = PlaneBase<float>;
    using Plane = Planef;

    float Distance(const Plane &plane, const Vector<float, 3> &point);

    float UnsignedDistance(const Plane &plane, const Vector<float, 3> &point);

    Vector<float, 3> ClosestPoint(const Plane &plane, const Vector<float, 3> &point);
}

#endif //ENGINE24_PLANE_H
