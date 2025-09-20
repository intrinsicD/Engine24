#pragma once

#include "GeometricTraits.h"
#include "AABB.h"
#include "Sphere.h"

namespace Bcg {
    template<typename T>
    struct IntersectsTraits<Sphere<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &a, const AABB<T> &b) noexcept {
            // Find the point on the AABB closest to the sphere center
            T x = std::max(b.min.x, std::min(a.center.x, b.max.x));
            T y = std::max(b.min.y, std::min(a.center.y, b.max.y));
            T z = std::max(b.min.z, std::min(a.center.z, b.max.z));

            // Calculate the squared distance from the sphere center to this point
            T distance_squared = (x - a.center.x) * (x - a.center.x) +
                                 (y - a.center.y) * (y - a.center.y) +
                                 (z - a.center.z) * (z - a.center.z);

            // Sphere and AABB intersect if the squared distance is less than or equal to the squared radius
            return distance_squared <= (a.radius * a.radius);
        }
    };

    template<typename T>
    struct IntersectsTraits<AABB<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool intersects(const AABB<T> &a, const Sphere<T> &b) noexcept {
            return IntersectsTraits<Sphere<T>, AABB<T> >::intersects(b, a);
        }
    };

    template<typename T>
    struct IntersectsTraits<Sphere<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &a, const AABB<T> &b) noexcept {
            // Find the point on the AABB closest to the sphere center
            T x = std::max(b.min.x, std::min(a.center.x, b.max.x));
            T y = std::max(b.min.y, std::min(a.center.y, b.max.y));
            T z = std::max(b.min.z, std::min(a.center.z, b.max.z));

            // Calculate the squared distance from the sphere center to this point
            T distance_squared = (x - a.center.x) * (x - a.center.x) +
                                 (y - a.center.y) * (y - a.center.y) +
                                 (z - a.center.z) * (z - a.center.z);

            // Sphere and AABB intersect if the squared distance is less than or equal to the squared radius
            return distance_squared <= (a.radius * a.radius);
        }
    };

    template<typename T>
    struct IntersectsTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool intersects(const AABB<T> &a, const Vector<T, 3> &b) noexcept {
            return a.min.x <= b.x && b.x <= a.max.x &&
                   a.min.y <= b.y && b.y <= a.max.y &&
                   a.min.z <= b.z && b.z <= a.max.z;
        }
    };

    template<typename T>
    struct IntersectsTraits<Vector<T, 3>, AABB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Vector<T, 3> &a, const AABB<T> &b) noexcept {
            return IntersectsTraits<AABB<T>, Vector<T, 3> >::intersects(b, a);
        }
    };

    template<typename T>
    struct IntersectsTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &a, const Vector<T, 3> &b) noexcept {
            return VecTraits<Vector<T, 3> >::squared_distance(a.center, b) <= a.radius * a.radius;
        }
    };

    template<typename T>
    struct IntersectsTraits<Vector<T, 3>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Vector<T, 3> &a, const Sphere<T> &b) noexcept {
            return IntersectsTraits<Sphere<T>, Vector<T, 3> >::intersects(b, a);
        }
    };
}
