#pragma once

#include "GeometricTraits.h"
#include "AABB.h"
#include "Sphere.h"

namespace Bcg {
    template<typename T>
    struct ContainsTraits<Sphere<T>, AABB<T> > {
        CUDA_HOST_DEVICE static bool contains(const Sphere<T> &a, const AABB<T> &b) noexcept {
            // Check if all 8 corners of the AABB are within the sphere
            Vector<T, 3> corners[8] = {
                {b.min.x, b.min.y, b.min.z},
                {b.min.x, b.min.y, b.max.z},
                {b.min.x, b.max.y, b.min.z},
                {b.min.x, b.max.y, b.max.z},
                {b.max.x, b.min.y, b.min.z},
                {b.max.x, b.min.y, b.max.z},
                {b.max.x, b.max.y, b.min.z},
                {b.max.x, b.max.y, b.max.z}
            };

            for (const auto &corner: corners) {
                T dist_squared = (corner.x - a.center.x) * (corner.x - a.center.x) +
                                 (corner.y - a.center.y) * (corner.y - a.center.y) +
                                 (corner.z - a.center.z) * (corner.z - a.center.z);
                if (dist_squared > (a.radius * a.radius)) {
                    return false; // At least one corner is outside the sphere
                }
            }
            return true; // All corners are inside the sphere
        }
    };

    template<typename T>
    struct ContainsTraits<AABB<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool contains(const AABB<T> &a, const Sphere<T> &b) noexcept {
            // Check if the sphere's center is within the AABB
            if (b.center.x - b.radius < a.min.x || b.center.x + b.radius > a.max.x ||
                b.center.y - b.radius < a.min.y || b.center.y + b.radius > a.max.y ||
                b.center.z - b.radius < a.min.z || b.center.z + b.radius > a.max.z) {
                return false; // Sphere extends outside the AABB
            }
            return true; // Sphere is fully contained within the AABB
        }
    };

    template<typename T>
    struct ContainsTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool contains(const Sphere<T> &a, const Vector<T, 3> &b) noexcept {
            return VecTraits<Vector<T, 3> >::squared_distance(a.center, b) <= a.radius * a.radius;
        }
    };
}
