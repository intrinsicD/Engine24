//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_OBB_H
#define ENGINE24_OBB_H

#include "Rotation.h"
#include "AABB.h"

namespace Bcg {
    template<typename T>
    struct OBBBase {
        Vector<T, 3> center;
        AngleAxis orientation;
        Vector<T, 3> half_extent;
    };

    using OBBf = OBBBase<float>;
    using OBB = OBBf;

    Vector<float, 3> ClosestPoint(const OBB &obb, const Vector<float, 3> &point);

    float Distance(const OBB &obb, const Vector<float, 3> &point);

    float UnsignedDistance(const OBB &obb, const Vector<float, 3> &point);
}

#endif //ENGINE24_OBB_H
