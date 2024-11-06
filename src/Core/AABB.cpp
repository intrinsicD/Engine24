//
// Created by alex on 30.10.24.
//

#include "AABB.h"

namespace Bcg {
    void Clear(AABB &aabb) {
        aabb.min = Vector<float, 3>(std::numeric_limits<float>::max());
        aabb.max = Vector<float, 3>(std::numeric_limits<float>::lowest());
    }

    void Grow(AABB &aabb, const Vector<float, 3> &point) {
        aabb.min = glm::min(aabb.min, point);
        aabb.max = glm::max(aabb.max, point);
    }

    void Build(AABB &aabb, const std::vector<Vector<float, 3>> &points) {
        Clear(aabb);
        for (const auto &point: points) {
            Grow(aabb, point);
        }
    }

    AABB Merge(const AABB &a, const AABB &b) {
        AABB result{};
        result.min = glm::min(a.min, b.min);
        result.max = glm::max(a.max, b.max);
        return result;
    }

    Vector<float, 3> Diagonal(const AABB &aabb) {
        return aabb.max - aabb.min;
    }

    Vector<float, 3> HalfExtent(const AABB &aabb) {
        return Diagonal(aabb) * 0.5f;
    }

    Vector<float, 3> Center(const AABB &aabb) {
        return (aabb.min + aabb.max) * 0.5f;
    }

    float Volume(const AABB &aabb) {
        return glm::compMul(Diagonal(aabb));
    }

    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point) {
        return glm::clamp(point, aabb.min, aabb.max);
    }

    bool Contains(const AABB &aabb, const Vector<float, 3> &point) {
        const glm::bvec3 cond1 = glm::greaterThanEqual(point, aabb.min);
        const glm::bvec3 cond2 = glm::greaterThanEqual(point, aabb.max);
        return glm::all(cond1) && glm::all(cond2);
    }

    bool Contains(const AABB &aabb, const AABB &other) {
        return Contains(aabb, other.min) && Contains(aabb, other.max);
    }

    bool Intersects(const AABB &a, const AABB &b) {
        const glm::bvec3 cond1 = glm::lessThanEqual(a.min, b.max);
        const glm::bvec3 cond2 = glm::greaterThanEqual(a.max, b.min);
        return glm::all(cond1) && glm::all(cond2);
    }

    AABB Intersection(const AABB &a, const AABB &b) {
        AABB result{};
        result.min = glm::max(a.min, b.min);
        result.max = glm::min(a.max, b.max);
        return result;
    }

    float Distance(const AABB &aabb, const Vector<float, 3> &point) {
        Vector<float, 3> d = ClosestPoint(aabb, point) - point;
        return glm::length(d);
    }
}
