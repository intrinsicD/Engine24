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
    
    void clear(AABB &aabb){
        aabb.min = glm::vec3(std::numeric_limits<float>::max());
        aabb.max = glm::vec3(std::numeric_limits<float>::lowest());
    }

    void grow(AABB &aabb, const glm::vec3 &point){
        aabb.min = glm::min(aabb.min, point);
        aabb.max = glm::max(aabb.max, point);
    }

    AABB merge(const AABB &a, const AABB &b){
        AABB result;
        result.min = glm::min(a.min, b.min);
        result.max = glm::max(a.max, b.max);
        return result;
    }

    glm::vec3 diagonal(const AABB &aabb){
        return aabb.max - aabb.min;
    }

    glm::vec3 half_extent(const AABB &aabb){
        return diagonal(aabb) / 2.0f;
    }

    glm::vec3 center(const AABB &aabb){
        return (aabb.min + aabb.max) / 2.0f;
    }

    float volume(const AABB &aabb){
        return glm::compMul(diagonal(aabb));
    }

    glm::vec3 closest_point(const AABB &aabb, const glm::vec3 &point){
        return glm::clamp(point, aabb.min, aabb.max);
    }

    bool contains(const AABB &aabb, const glm::vec3 &point){
        return glm::all(glm::greaterThanEqual(point, aabb.min)) && glm::all(glm::lessThanEqual(point, aabb.max));
    }

    bool contains(const AABB &aabb, const AABB &other){
        return contains(aabb, other.min) && contains(aabb, other.max);
    }

    bool intersects(const AABB &a, const AABB &b){
        return glm::all(glm::lessThanEqual(a.min, b.max)) && glm::all(glm::greaterThanEqual(a.max, b.min));
    }

    AABB intersection(const AABB &a, const AABB &b){
        AABB result;
        result.min = glm::max(a.min, b.min);
        result.max = glm::min(a.max, b.max);
        return result;
    }
    
    float distance(const AABB &aabb, const glm::vec3 &point) {
        glm::vec3 d = closest_point(aabb, point) - point;
        return glm::length(d);
    }
}

#endif //ENGINE24_AABB_H
