//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"
#include "StringTraits.h"
#include "VecTraits.h"
#include "Macros.h"
#include "GeometricTraits.h"

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

    //TODO define free functions to make use of AABBBase<T> with GeometricTraits.h easier for a concise API

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

    //------------------------------------------------------------------------------------------------------------------

    template<typename T>
    struct ClosestPointTraits<AABBBase<T>, Vector<T, 3>> {
        CUDA_HOST_DEVICE
        static Vector<T, 3> closest_point(const AABBBase<T> &aabb, const Vector<T, 3> &point) noexcept {
            return VecTraits<Vector<T, 3> >::clamp(point, aabb.min, aabb.max);
        }
    };

    template<typename T>
    struct SquaredDistanceTraits<AABBBase<T>, Vector<T, 3>> {
        CUDA_HOST_DEVICE
        static T squared_distance(const AABBBase<T> &aabb, const Vector<T, 3> &point) noexcept {
            return VecTraits<Vector<T, 3> >::squared_distance(
                    ClosestPointTraits<AABBBase<T>, Vector<T, 3>>::closest_point(aabb, point), point);
        }
    };

    template<typename T>
    struct DistanceTraits<AABBBase<T>, Vector<T, 3>> {
        CUDA_HOST_DEVICE
        static T distance(const AABBBase<T> &aabb, const Vector<T, 3> &point) noexcept {
            return sqrt(SquaredDistanceTraits<AABBBase<T>, Vector<T, 3>>::squared_distance(aabb, point));
        }
    };

    template<typename T>
    struct GetterTraits<AABBBase<T>, Vector<T, 3>> {
        CUDA_HOST_DEVICE
        static AABBBase<T> getter(const Vector<T, 3> &v) noexcept {
            return {v, v};
        }
    };

    template<typename T>
    CUDA_HOST_DEVICE
    static bool isWithinBounds(const AABBBase<T> &a, const Vector<T, 3> &b) noexcept {
        return a.min.x <= b.x && b.x <= a.max.x &&
               a.min.y <= b.y && b.y <= a.max.y &&
               a.min.z <= b.z && b.z <= a.max.z;
    }

    template<typename T>
    struct ContainsTraits<AABBBase<T>, Vector<T, 3>> {
        CUDA_HOST_DEVICE
        static bool contains(const AABBBase<T> &a, const Vector<T, 3> &b) noexcept {
            return isWithinBounds(a, b);
        }
    };

    template<typename T>
    struct IntersectsTraits<AABBBase<T>, Vector<T, 3>> {
        CUDA_HOST_DEVICE
        static bool intersects(const AABBBase<T> &a, const Vector<T, 3> &b) noexcept {
            return isWithinBounds(a, b);
        }
    };

    template<typename T>
    struct ContainsTraits<AABBBase<T>, AABBBase<T>> {
        CUDA_HOST_DEVICE
        static bool contains(const AABBBase<T> &a, const AABBBase<T> &b) noexcept {
            return a.min.x <= b.min.x && b.max.x <= a.max.x &&
                   a.min.y <= b.min.y && b.max.y <= a.max.y &&
                   a.min.z <= b.min.z && b.max.z <= a.max.z;
        }
    };

    template<typename T>
    struct IntersectsTraits<AABBBase<T>, AABBBase<T>> {
        CUDA_HOST_DEVICE
        static bool intersects(const AABBBase<T> &a, const AABBBase<T> &b) noexcept {
            return !(a.max.x < b.min.x || b.max.x < a.min.x ||
                     a.max.y < b.min.y || b.max.y < a.min.y ||
                     a.max.z < b.min.z || b.max.z < a.min.z);
        }
    };

    template<typename T>
    struct IntersectionTraits<AABBBase<T>, AABBBase<T>> {
        CUDA_HOST_DEVICE
        static AABBBase<T> intersection(const AABBBase<T> &a, const AABBBase<T> &b) noexcept {
            AABBBase<T> result{};
            result.min = VecTraits<Vector<T, 3> >::cwiseMax(a.min, b.min);
            result.max = VecTraits<Vector<T, 3> >::cwiseMin(a.max, b.max);
            return result;
        }
    };


}

#endif //ENGINE24_AABB_H
