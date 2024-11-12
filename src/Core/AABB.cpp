//
// Created by alex on 30.10.24.
//

#include "AABB.h"
#include "GlmToEigen.h"

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
        return AABB::diagonal(aabb.min, aabb.max);
    }

    Vector<float, 3> HalfExtent(const AABB &aabb) {
        return AABB::half_extent(aabb.min, aabb.max);
    }

    Vector<float, 3> Center(const AABB &aabb) {
        return AABB::center(aabb.min, aabb.max);
    }

    float Volume(const AABB &aabb) {
        return AABB::volume(aabb.min, aabb.max);
    }

    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point) {
        return AABB::closest_point(aabb.min, aabb.max, point);
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
        return AABB::distance(aabb.min, aabb.max, point);
    }

    std::string StringTraits<AABB>::ToString(const Bcg::AABB &aabb) {
        std::stringstream ss;
        ss << MapConst(aabb.min).transpose() << " " << MapConst(aabb.max).transpose();
        return ss.str();
    }
}
