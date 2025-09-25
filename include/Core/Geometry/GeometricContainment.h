#pragma once

#include "GeometricTraits.h"
#include "AABB.h"
#include "Sphere.h"
#include "OBB.h"

namespace Bcg {
    template<typename T>
    struct ContainsTraits<Sphere<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool contains(const Sphere<T> &sphere, const AABB<T> &aabb) noexcept {
            assert(aabb.min.x <= aabb.max.x && aabb.min.y <= aabb.max.y && aabb.min.z <= aabb.max.z);
            assert(sphere.radius >= T(0));
            // farthest distance from s.center to the AABB occurs at the corner with
            // per-axis farthest endpoint from the center.
            const T dx = (sphere.center.x < (aabb.min.x + aabb.max.x) * T(0.5))
                             ? (aabb.max.x - sphere.center.x)
                             : (sphere.center.x - aabb.min.x);
            const T dy = (sphere.center.y < (aabb.min.y + aabb.max.y) * T(0.5))
                             ? (aabb.max.y - sphere.center.y)
                             : (sphere.center.y - aabb.min.y);
            const T dz = (sphere.center.z < (aabb.min.z + aabb.max.z) * T(0.5))
                             ? (aabb.max.z - sphere.center.z)
                             : (sphere.center.z - aabb.min.z);
            const T d2 = dx * dx + dy * dy + dz * dz;
            return d2 <= sphere.radius * sphere.radius; // touching counts as contained
        }
    };

    template<typename T>
    struct ContainsTraits<AABB<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool contains(const AABB<T> &aabb, const Sphere<T> &sphere) noexcept {
            assert(aabb.min.x <= aabb.max.x && aabb.min.y <= aabb.max.y && aabb.min.z <= aabb.max.z);
            assert(sphere.radius >= T(0));
            // AABB contains Sphere if the sphere is fully inside the AABB
            return !(sphere.center.x - sphere.radius < aabb.min.x ||
                     sphere.center.x + sphere.radius > aabb.max.x ||
                     sphere.center.y - sphere.radius < aabb.min.y ||
                     sphere.center.y + sphere.radius > aabb.max.y ||
                     sphere.center.z - sphere.radius < aabb.min.z ||
                     sphere.center.z + sphere.radius > aabb.max.z);
        }
    };
}
