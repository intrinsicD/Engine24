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
    struct AABB {
        Vector<T, 3> min = Vector<T, 3>(std::numeric_limits<T>::max());
        Vector<T, 3> max = Vector<T, 3>(std::numeric_limits<T>::lowest());

        CUDA_HOST_DEVICE AABB() = default;

        CUDA_HOST_DEVICE explicit AABB(const Vector<T, 3> &min, const Vector<T, 3> &max) : min(min), max(max) {
        }

        CUDA_HOST_DEVICE explicit AABB(const Vector<T, 3> &point) : min(point), max(point) {
        }

        CUDA_HOST_DEVICE bool is_valid() const {
            return min.x <= max.x && min.y <= max.y && min.z <= max.z;
        }

        template<typename Iterator>
        CUDA_HOST static AABB Build(const Iterator &begin, const Iterator &end) {
            AABB result;
            for (auto it = begin; it != end; ++it) {
                result.grow(*it);
            }
            return result;
        }

        CUDA_HOST_DEVICE void clear() {
            min = Vector<T, 3>(std::numeric_limits<T>::max());
            max = Vector<T, 3>(std::numeric_limits<T>::lowest());
        }

        CUDA_HOST_DEVICE void merge(const AABB &other) {
            min = VecTraits<Vector<T, 3> >::cwiseMin(min, other.min);
            max = VecTraits<Vector<T, 3> >::cwiseMax(max, other.max);
        }

        CUDA_HOST_DEVICE void grow(const Vector<T, 3> &point) {
            min = VecTraits<Vector<T, 3> >::cwiseMin(min, point);
            max = VecTraits<Vector<T, 3> >::cwiseMax(max, point);
        }

        CUDA_HOST_DEVICE void grow(const AABB &aabb) {
            merge(aabb);
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

        CUDA_HOST_DEVICE T surface_area() const {
            Vector<T, 3> d = diagonal();
            return T(2) * (d.x * d.y + d.y * d.z + d.z * d.x);
        }
    };

    template<typename T>
    struct StringTraits<AABB<T> > {
        static std::string ToString(const AABB<T> &aabb) {
            std::stringstream ss;
            ss << MapConst(aabb.min).transpose() << " " << MapConst(aabb.max).transpose();
            return ss.str();
        }
    };

    //------------------------------------------------------------------------------------------------------------------

    template<typename T>
    struct ClosestPointTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static Vector<T, 3> closest_point(const AABB<T> &aabb, const Vector<T, 3> &point) noexcept {
            return VecTraits<Vector<T, 3> >::clamp(point, aabb.min, aabb.max);
        }
    };

    template<typename T>
    struct SquaredDistanceTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static T squared_distance(const AABB<T> &aabb, const Vector<T, 3> &point) noexcept {
            return VecTraits<Vector<T, 3> >::squared_distance(
                ClosestPointTraits<AABB<T>, Vector<T, 3> >::closest_point(aabb, point), point);
        }
    };

    template<typename T>
    struct DistanceTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static T distance(const AABB<T> &aabb, const Vector<T, 3> &point) noexcept {
            return sqrt(SquaredDistanceTraits<AABB<T>, Vector<T, 3> >::squared_distance(aabb, point));
        }
    };

    template<typename T>
    struct BuilderTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static AABB<T> build(const Vector<T, 3> &v) noexcept {
            return {v, v};
        }
    };

    template<typename T>
    CUDA_HOST_DEVICE static bool isWithinBounds(const AABB<T> &a, const Vector<T, 3> &b) noexcept {
        return a.min.x <= b.x && b.x <= a.max.x &&
               a.min.y <= b.y && b.y <= a.max.y &&
               a.min.z <= b.z && b.z <= a.max.z;
    }

    template<typename T>
    struct ContainsTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool contains(const AABB<T> &a, const Vector<T, 3> &b) noexcept {
            return isWithinBounds(a, b);
        }
    };

    template<typename T>
    struct IntersectsTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool intersects(const AABB<T> &a, const Vector<T, 3> &b) noexcept {
            return isWithinBounds(a, b);
        }
    };

    template<typename T>
    struct ContainsTraits<AABB<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool contains(const AABB<T> &a, const AABB<T> &b) noexcept {
            return a.min.x <= b.min.x && b.max.x <= a.max.x &&
                   a.min.y <= b.min.y && b.max.y <= a.max.y &&
                   a.min.z <= b.min.z && b.max.z <= a.max.z;
        }
    };

    template<typename T>
    struct IntersectsTraits<AABB<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const AABB<T> &a, const AABB<T> &b) noexcept {
            return !(a.max.x < b.min.x || b.max.x < a.min.x ||
                     a.max.y < b.min.y || b.max.y < a.min.y ||
                     a.max.z < b.min.z || b.max.z < a.min.z);
        }
    };

    template<typename T>
    struct IntersectionTraits<AABB<T>, AABB<T> > {
        CUDA_HOST_DEVICE static AABB<T> intersection(const AABB<T> &a, const AABB<T> &b) noexcept {
            AABB<T> result{};
            result.min = VecTraits<Vector<T, 3> >::cwiseMax(a.min, b.min);
            result.max = VecTraits<Vector<T, 3> >::cwiseMin(a.max, b.max);
            return result;
        }
    };
}

#endif //ENGINE24_AABB_H
