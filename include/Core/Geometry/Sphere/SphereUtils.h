//
// Created by alex on 6/3/25.
//

#ifndef SPHEREUTILS_H
#define SPHEREUTILS_H

#include "Sphere.h"
#include "AABB.h"
#include "OBB.h"

namespace Bcg {
    template<typename T, typename Shape>
    struct BuilderTraits<Sphere<T>, Shape> {
        CUDA_HOST_DEVICE static Sphere<T> build(const Shape &) noexcept {
            static_assert(sizeof(Shape) == 0, "BuilderTraits<Sphere, Shape> not implemented for this shape.");
            return Sphere<T>();
        }

        CUDA_HOST_DEVICE static Sphere<T> build(const OBB<T> &shape) noexcept {
            return Sphere<T>{shape.center, glm::compMax(shape.half_extents)};
        }

        CUDA_HOST_DEVICE static Sphere<T> build(const AABB<T> &shape) noexcept {
            return Sphere<T>{shape.center(), glm::compMax(shape.half_extent())};
        }

        CUDA_HOST_DEVICE static Sphere<T> build(const Vector<T, 3> &shape) noexcept {
            return Sphere<T>{shape, 0};
        }

        CUDA_HOST static Sphere<T> build(const std::vector<Vector<T, 3> > &points) noexcept {
            if (points.empty()) return Sphere<T>();
            Vector<T, 3> c(0);
            for (const auto &p: points) c += p;
            c /= static_cast<T>(points.size());
            T r2 = T(0);
            for (const auto &p: points) {
                const Vector<T, 3> d = p - c;
                r2 = glm::max(r2, glm::dot(d, d));
            }
            const T r = glm::sqrt(r2);
            return Sphere<T>{c, r};
        }

        CUDA_HOST static std::vector<Sphere<T>> build_all(const std::vector<Vector<T, 3> > &points, T r) noexcept {
            std::vector<Sphere<T>> spheres;
            for (const auto &p: points) {
                spheres.emplace_back(p, r);
            }
            return spheres;
        }
    };

    template<typename T>
    Vector<T, 3> ClosestPoint(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return ClosestPointTraits<Sphere<T>, Vector<T, 3> >::closest_point(sphere, point);
    }

    template<typename T>
    T SquaredDistance(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return SquaredDistanceTraits<Sphere<T>, Vector<T, 3> >::squared_distance(sphere, point);
    }

    template<typename T>
    T Distance(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return DistanceTraits<Sphere<T>, Vector<T, 3> >::distance(sphere, point);
    }

    template<typename T>
    bool Contains(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return ContainsTraits<Sphere<T>, Vector<T, 3> >::contains(sphere, point);
    }

    template<typename T>
    bool Intersects(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return IntersectsTraits<Sphere<T>, Vector<T, 3> >::intersects(sphere, point);
    }

    template<typename T>
    bool Contains(const Sphere<T> &a, const Sphere<T> &b) {
        return ContainsTraits<Sphere<T>, Sphere<T> >::contains(a, b);
    }

    template<typename T>
    bool Intersects(const Sphere<T> &a, const Sphere<T> &b) {
        return IntersectsTraits<Sphere<T>, Sphere<T> >::intersects(a, b);
    }
}
#endif //SPHEREUTILS_H
