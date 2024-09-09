//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"
#include <limits>

namespace Bcg {
    template<typename T>
    struct AABBbase {
        using Scalar = T;
        Vector<Scalar, 3> min;
        Vector<Scalar, 3> max;

        AABBbase() : min(Vector<Scalar, 3>::Constant(std::numeric_limits<Scalar>::max())),
                     max(Vector<Scalar, 3>::Constant(-std::numeric_limits<Scalar>::max())) {}

        template<typename InputIterator>
        AABBbase(InputIterator first, InputIterator last) {
            for (auto it = first; it != last; ++it) {
                grow(*it);
            }
        }

        explicit AABBbase(const std::vector<Vector<Scalar, 3>> &points) : AABBbase(points.begin(), points.end()) {}

        inline void grow(const Vector<Scalar, 3> &point) {
            min = min.cwiseMin(point);
            max = max.cwiseMax(point);
        }

        Vector<Scalar, 3> diagonal() const {
            return max - min;
        }

        Vector<Scalar, 3> half_extent() const {
            return diagonal() / 2;
        }

        Vector<Scalar, 3> center() const {
            return (min + max) / 2;
        }

        static Vector<Scalar, 3>
        closest_point(const Vector<Scalar, 3> &min, const Vector<Scalar, 3> &max, const Vector<Scalar, 3> &point) {
            return point.cwiseMax(min).cwiseMin(max);
        }
    };

    using AABBf = AABBbase<float>;
    using AABB = AABBf;

    AABB &Grow(AABB &aabb, const Vector<float, 3> &point);

    AABB &Build(AABB &aabb, const std::vector<Vector<float, 3>> &points);

    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point);

    float Distance(const AABB &aabb, const Vector<float, 3> &point);

    float UnsigendDistance(const AABB &aabb, const Vector<float, 3> &point);
}

#endif //ENGINE24_AABB_H
