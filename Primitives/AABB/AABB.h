//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "../MatVec.h"
#include <limits>

namespace Bcg {
    template<typename T>
    struct AABB {
        Vector<float, 3> min = Vector<float, 3>::Constant(std::numeric_limits<float>::max());
        Vector<float, 3> max = Vector<float, 3>::Constant(-std::numeric_limits<float>::max());

        Vector<T, 3> center() const {
            return (min + max) / 2;
        }

        static Vector<T, 3> closest_point(const Vector<T, 3> &min, const Vector<T, 3> &max, const Vector<T, 3> &point) {
            return point.cwiseMax(min).cwiseMin(max);
        }
    };

    template<typename T>
    AABB<T> &Grow(AABB<T> &aabb, const Vector<T, 3> &point) {
        aabb.min = aabb.min.cwiseMin(point);
        aabb.max = aabb.max.cwiseMax(point);
        return aabb;
    }

    template<typename T>
    AABB<T> &Build(AABB<T> &aabb, const std::vector<Vector<T, 3>> &points) {
        aabb = AABB<T>();
        for(auto &point : points){
            Grow(aabb, point);
        }
        return aabb;
    }

    template<typename T>
    Vector<T, 3> closest_point(const AABB<T> &aabb, const Vector<T, 3> &point) {
        return AABB<T>::closest_point(aabb.min, aabb.max, point);
    }

    template<typename T>
    T distance(const AABB<T> &aabb, const Vector<T, 3> &point) {
        return (point - closest_point(aabb, point)).norm();
    }

    template<typename T>
    T unsigned_distance(const AABB<T> &aabb, const Vector<T, 3> &point) {
        return std::abs(AABB<T>::distance(aabb, point));
    }
}

#endif //ENGINE24_AABB_H
