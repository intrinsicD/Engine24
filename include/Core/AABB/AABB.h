//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    struct AABBBase{
        Vector<T, 3> min;
        Vector<T, 3> max;
    };

    using AABBf = AABBBase<float>;
    using AABB = AABBf;

    void Clear(AABB &aabb);

    void Grow(AABB &aabb, const Vector<float, 3> &point);

    void Build(AABB &aabb, const std::vector<Vector<float, 3>> &points);

    AABB Merge(const AABB &a, const AABB &b);

    Vector<float, 3> Diagonal(const AABB &aabb);

    Vector<float, 3> HalfExtent(const AABB &aabb);

    Vector<float, 3> Center(const AABB &aabb);

    float Volume(const AABB &aabb);

    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point);

    bool Contains(const AABB &aabb, const Vector<float, 3> &point);

    bool Contains(const AABB &aabb, const AABB &other);

    bool Intersects(const AABB &a, const AABB &b);

    AABB Intersection(const AABB &a, const AABB &b);
    
    float Distance(const AABB &aabb, const Vector<float, 3> &point);
}

#endif //ENGINE24_AABB_H
