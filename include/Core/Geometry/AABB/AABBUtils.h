//
// Created by alex on 11/17/24.
//

#ifndef AABBUTILS_H
#define AABBUTILS_H

#include "AABB.h"

namespace Bcg::AABBUtils {
    template<typename T>
    std::vector<AABB<T> > ConvertToAABBs(const std::vector<Vector<T, 3> > &points) {
        if (points.empty()) return {};
        std::vector<AABB<T> > aabbs;
        aabbs.reserve(points.size());
        for (const auto &p: points) {
            aabbs.emplace_back(p);
        }
        return aabbs;
    }

    template<typename T>
    Vector<float, 3> ClosestPoint(const AABB<T> &aabb, const Vector<float, 3> &point) {
        return ClosestPointTraits<AABB<T>, Vector<T, 3> >::closest_point(aabb, point);
    }

    template<typename T>
    bool Contains(const AABB<T> &aabb, const Vector<float, 3> &point) {
        return ContainsTraits<AABB<T>, Vector<T, 3> >::contains(aabb, point);
    }

    template<typename T>
    bool Contains(const AABB<T> &aabb, const AABB<T> &other) {
        return ContainsTraits<AABB<T>, AABB<T> >::contains(aabb, other);
    }

    template<typename T>
    bool Intersects(const AABB<T> &a, const AABB<T> &b) {
        return IntersectsTraits<AABB<T>, AABB<T> >::intersects(a, b);
    }

    template<typename T>
    AABB<T> Intersection(const AABB<T> &a, const AABB<T> &b) {
        return IntersectionTraits<AABB<T>, AABB<T> >::intersects(a, b);
    }

    template<typename T>
    float Distance(const AABB<T> &aabb, const Vector<float, 3> &point) {
        return DistanceTraits<AABB<T>, AABB<T> >::distance(aabb, point);
    }
}

#endif //AABBUTILS_H
