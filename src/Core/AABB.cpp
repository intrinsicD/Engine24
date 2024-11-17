//
// Created by alex on 30.10.24.
//

#include "AABB.h"
#include "AABBUtils.h"
#include "GlmToEigen.h"
#include "VecTraits.h"

namespace Bcg {
    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point) {
        return AABBUtils::closest_point(aabb.min, aabb.max, point);
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
        return AABBUtils::distance(aabb.min, aabb.max, point);
    }

    std::string StringTraits<AABB>::ToString(const Bcg::AABB &aabb) {
        std::stringstream ss;
        ss << MapConst(aabb.min).transpose() << " " << MapConst(aabb.max).transpose();
        return ss.str();
    }
}
