//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"

namespace Bcg {
    struct AABB{
        glm::vec3 min;
        glm::vec3 max;
    };
    
    void Clear(AABB &aabb);

    void Grow(AABB &aabb, const glm::vec3 &point);

    void Build(AABB &aabb, const std::vector<glm::vec3> &points);

    AABB Merge(const AABB &a, const AABB &b);

    glm::vec3 Diagonal(const AABB &aabb);

    glm::vec3 HalfExtent(const AABB &aabb);

    glm::vec3 Center(const AABB &aabb);

    float Volume(const AABB &aabb);

    glm::vec3 ClosestPoint(const AABB &aabb, const glm::vec3 &point);

    bool Contains(const AABB &aabb, const glm::vec3 &point);

    bool Contains(const AABB &aabb, const AABB &other);

    bool Intersects(const AABB &a, const AABB &b);

    AABB Intersection(const AABB &a, const AABB &b);
    
    float Distance(const AABB &aabb, const glm::vec3 &point);
}

#endif //ENGINE24_AABB_H
