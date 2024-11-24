//
// Created by alex on 11/17/24.
//

#ifndef AABBUTILS_H
#define AABBUTILS_H

#include "MatVec.h"

namespace Bcg::AABBUtils {
    template<typename T>
    Vector<T, 3> diagonal(const Vector<T, 3> &min, const Vector<T, 3> &max) {
        return max - min;
    }

    template<typename T>
    Vector<T, 3> half_extent(const Vector<T, 3> &min, const Vector<T, 3> &max) {
        return diagonal(min, max) * 0.5f;
    }

    template<typename T>
    Vector<T, 3> center(const Vector<T, 3> &min, const Vector<T, 3> &max) {
        return (min + max) * 0.5f;
    }

    template<typename T>
    Vector<T, 3> closest_point(const Vector<T, 3> &min, const Vector<T, 3> &max, const Vector<T, 3> &point) {
        return glm::clamp(point, min, max);
    }

    template<typename T>
    T distance(const Vector<T, 3> &min, const Vector<T, 3> &max, const Vector<T, 3> &point) {
        return glm::length(closest_point(min, max, point));
    }

    template<typename T>
    T volume(const Vector<T, 3> &min, const Vector<T, 3> &max) {
        return glm::compMul(diagonal(min, max));
    }
}

#endif //AABBUTILS_H
