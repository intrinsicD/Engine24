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
        CUDA_HOST_DEVICE static Sphere<T> build(const OBB<T> &shape) noexcept {
            // radius should be distance to a corner = length of half_extents
            return Sphere<T>{shape.center, glm::length(shape.half_extents)};
        }

        CUDA_HOST_DEVICE static Sphere<T> build(const AABB<T> &shape) noexcept {
            // radius should be distance to a corner = length of half_extent()
            return Sphere<T>{shape.center(), glm::length(shape.half_extent())};
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

        // Builder for Sphere from a vector of spheres: iterative minimal enclosing union of pairs
        CUDA_HOST static Sphere<T> build(const std::vector<Sphere<T> > &spheres) noexcept {
            Sphere<T> res{};
            if (spheres.empty()) return res;
            // Initialize with the first sphere (ensure non-negative radius)
            res.center = spheres[0].center;
            res.radius = glm::abs(spheres[0].radius);
            for (size_t i = 1; i < spheres.size(); ++i) {
                Sphere<T> s = spheres[i];
                s.radius = glm::abs(s.radius);
                // Check containment cases
                const Vector<T, 3> d = s.center - res.center;
                const T dist2 = glm::dot(d, d);
                const T dist = glm::sqrt(dist2);
                if (res.radius >= s.radius + dist) {
                    // s is inside res; nothing to do
                    continue;
                }
                if (s.radius >= res.radius + dist) {
                    // res is inside s; take s
                    res = s;
                    continue;
                }
                // General case: compute minimal enclosing sphere of two spheres
                const T new_r = (dist + res.radius + s.radius) * T(0.5);
                Vector<T, 3> new_c = res.center;
                if (dist > T(0)) {
                    const T t = (new_r - res.radius) / dist;
                    new_c = res.center + d * t;
                }
                res.center = new_c;
                res.radius = new_r;
            }
            return res;
        }

        // Build from many AABBs: convert to spheres and merge
        CUDA_HOST static Sphere<T> build(const std::vector<AABB<T> > &aabbs) noexcept {
            if (aabbs.empty()) return Sphere<T>();
            std::vector<Sphere<T> > spheres;
            spheres.reserve(aabbs.size());
            for (const auto &a: aabbs) {
                spheres.emplace_back(build(a));
            }
            return build(spheres);
        }

        // Build from many OBBs: convert to spheres and merge
        CUDA_HOST static Sphere<T> build(const std::vector<OBB<T> > &obbs) noexcept {
            if (obbs.empty()) return Sphere<T>();
            std::vector<Sphere<T> > spheres;
            spheres.reserve(obbs.size());
            for (const auto &o: obbs) {
                spheres.emplace_back(build(o));
            }
            return build(spheres);
        }

        CUDA_HOST static std::vector<Sphere<T> > build_all(const std::vector<Vector<T, 3> > &points, T r) noexcept {
            std::vector<Sphere<T> > spheres;
            for (const auto &p: points) {
                spheres.emplace_back(p, r);
            }
            return spheres;
        }
    };

    template<typename T, typename Shapes>
    struct BuilderTraits<std::vector<Sphere<T> >, Shapes> {
        CUDA_HOST static std::vector<Sphere<T> > build(const std::vector<Vector<T, 3>> &shapes, T r = T(0)) noexcept {
            std::vector<Sphere<T> > spheres;
            for (const auto &s: shapes) {
                spheres.emplace_back(s, r);
            }
            return spheres;
        }

        CUDA_HOST static std::vector<Sphere<T> > build(const std::vector<AABB<T>> &shapes, T r = T(0)) noexcept {
            std::vector<Sphere<T> > spheres;
            for (const auto &s: shapes) {
                const auto sphere = BuilderTraits<Sphere<T>, AABB<T>>::build(s);
                sphere.radius = glm::max(sphere.radius, r);
                spheres.emplace_back(sphere);
            }
            return spheres;
        }

        CUDA_HOST static std::vector<Sphere<T> > build(const std::vector<OBB<T>> &shapes, T r = T(0)) noexcept {
            std::vector<Sphere<T> > spheres;
            for (const auto &s: shapes) {
                const auto sphere = BuilderTraits<Sphere<T>, OBB<T>>::build(s);
                sphere.radius = glm::max(sphere.radius, r);
                spheres.emplace_back(sphere);
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
