//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"
#include <limits>

namespace Bcg {
    struct AABB{
        glm::vec3 min;
        glm::vec3 max;
    };
    
    void clear(AABB &aabb);

    void grow(AABB &aabb, const glm::vec3 &point);

    void build(AABB &aabb, const std::vector<glm::vec3> &points);

    AABB merge(const AABB &a, const AABB &b);

    glm::vec3 diagonal(const AABB &aabb);

    glm::vec3 half_extent(const AABB &aabb);

    glm::vec3 center(const AABB &aabb);

    float volume(const AABB &aabb);

    glm::vec3 closest_point(const AABB &aabb, const glm::vec3 &point);

    bool contains(const AABB &aabb, const glm::vec3 &point);

    bool contains(const AABB &aabb, const AABB &other);

    bool intersects(const AABB &a, const AABB &b);

    AABB intersection(const AABB &a, const AABB &b);
    
    float distance(const AABB &aabb, const glm::vec3 &point);
}

#endif //ENGINE24_AABB_H
