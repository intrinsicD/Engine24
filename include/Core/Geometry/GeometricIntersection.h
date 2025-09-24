#pragma once

#include "GeometricTraits.h"
#include "MathUtils.h"
#include "AABB.h"
#include "Sphere.h"

namespace Bcg {
    template<typename T>
    struct IntersectsTraits<Sphere<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &s, const AABB<T> &a) noexcept {
            assert(a.min.x <= a.max.x && a.min.y <= a.max.y && a.min.z <= a.max.z);
            assert(s.radius >= T(0));
            //Branchless distance (tiny speed win): avoids the clamp and uses only max operations (often faster on GPU/CPU)
            const T dx = d_max(d_max(a.min.x - s.center.x, T(0)), s.center.x - a.max.x);
            const T dy = d_max(d_max(a.min.y - s.center.y, T(0)), s.center.y - a.max.y);
            const T dz = d_max(d_max(a.min.z - s.center.z, T(0)), s.center.z - a.max.z);
            const T d2 = dx * dx + dy * dy + dz * dz;
            return d2 <= s.radius * s.radius;
        }
    };

    template<typename T>
    struct IntersectsTraits<AABB<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool intersects(const AABB<T> &a, const Sphere<T> &s) noexcept {
            return IntersectsTraits<Sphere<T>, AABB<T> >::intersects(s, a);
        }
    };

    template<typename T>
    struct IntersectsTraits<AABB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool intersects(const AABB<T> &a, const Vector<T, 3> &p) noexcept {
            assert(a.min.x <= a.max.x && a.min.y <= a.max.y && a.min.z <= a.max.z);
            return a.min.x <= p.x && p.x <= a.max.x &&
                   a.min.y <= p.y && p.y <= a.max.y &&
                   a.min.z <= p.z && p.z <= a.max.z;
        }
    };

    template<typename T>
    struct IntersectsTraits<Vector<T, 3>, AABB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Vector<T, 3> &p, const AABB<T> &a) noexcept {
            return IntersectsTraits<AABB<T>, Vector<T, 3> >::intersects(p, a);
        }
    };

    template<typename T>
    struct IntersectsTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &s, const Vector<T, 3> &b) noexcept {
            return VecTraits<Vector<T, 3> >::squared_distance(s.center, b) <= s.radius * s.radius;
        }
    };

    template<typename T>
    struct IntersectsTraits<Vector<T, 3>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Vector<T, 3> &a, const Sphere<T> &s) noexcept {
            return IntersectsTraits<Sphere<T>, Vector<T, 3> >::intersects(s, a);
        }
    };
}
