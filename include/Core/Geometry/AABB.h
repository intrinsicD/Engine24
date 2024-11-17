//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"
#include "StringTraits.h"
#include "VecTraits.h"
#include "Macros.h"

namespace Bcg {
    template<typename T>
    struct AABBBase {
        Vector<T, 3> min = Vector<T, 3>(std::numeric_limits<T>::max());
        Vector<T, 3> max = Vector<T, 3>(std::numeric_limits<T>::lowest());

        CUDA_HOST_DEVICE static AABBBase FromPoint(const Vector<T, 3> &point) {
            return {point, point};
        }

        template<typename Iterator>
        CUDA_HOST static AABBBase Build(const Iterator &begin, const Iterator &end) {
            AABBBase result;
            for (auto it = begin; it != end; ++it) {
                result.grow(*it);
            }
            return result;
        }

        CUDA_HOST_DEVICE void clear() {
            min = Vector<T, 3>(std::numeric_limits<T>::max());
            max = Vector<T, 3>(std::numeric_limits<T>::lowest());
        }

        CUDA_HOST_DEVICE void merge(const AABBBase &other) {
            min = VecTraits<Vector<T, 3> >::cwiseMin(min, other.min);
            max = VecTraits<Vector<T, 3> >::cwiseMax(max, other.max);
        }

        CUDA_HOST_DEVICE void grow(const Vector<T, 3> &point) {
            min = VecTraits<Vector<T, 3> >::cwiseMin(min, point);
            max = VecTraits<Vector<T, 3> >::cwiseMax(max, point);
        }

        CUDA_HOST_DEVICE Vector<T, 3> diagonal() const {
            return max - min;
        }

        CUDA_HOST_DEVICE Vector<T, 3> half_extent() const {
            return diagonal() * 0.5f;
        }

        CUDA_HOST_DEVICE Vector<T, 3> center() const {
            return (min + max) * 0.5f;
        }

        CUDA_HOST_DEVICE T volume() const {
            return VecTraits<Vector<T, 3> >::prod(diagonal());
        }
    };

    using AABBf = AABBBase<float>;
    using AABB = AABBf;

    Vector<float, 3> ClosestPoint(const AABB &aabb, const Vector<float, 3> &point);

    bool Contains(const AABB &aabb, const Vector<float, 3> &point);

    bool Contains(const AABB &aabb, const AABB &other);

    bool Intersects(const AABB &a, const AABB &b);

    AABB Intersection(const AABB &a, const AABB &b);

    float Distance(const AABB &aabb, const Vector<float, 3> &point);

    template<>
    struct StringTraits<AABB> {
        static std::string ToString(const AABB &aabb);
    };
}

#endif //ENGINE24_AABB_H
