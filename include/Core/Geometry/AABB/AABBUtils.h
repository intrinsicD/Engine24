//
// Created by alex on 11/17/24.
//

#ifndef AABBUTILS_H
#define AABBUTILS_H

#include "AABB.h"
#include "OBBUtils.h"
#include "GeometricTraits.h"

namespace Bcg {
    template<typename T>
    std::vector<AABB<T> > ConvertToAABBs(const std::vector<Vector<T, 3> > &points) {
        if (points.empty()) return {};
        std::vector<AABB<T> > aabbs;
        aabbs.reserve(points.size());
        for (const auto &p: points) {
            aabbs.emplace_back(p);
        }
        return aabbs;
    }

    template<typename T>
    std::array<Vector<T, 3>, 8> GetVertices(const AABB<T> &aabb) {
        std::array<Vector<T, 3>, 8> v{};
        for (int i = 0; i < 8; ++i) {
            v[i] = Vector<T, 3>{
                (i & 1) ? aabb.max.x : aabb.min.x,
                (i & 2) ? aabb.max.y : aabb.min.y,
                (i & 4) ? aabb.max.z : aabb.min.z
            };
        }
        return v;
    }


    template<typename T>
    constexpr std::array<std::array<int, 4>, 6> GetFaceQuads(const AABB<T> &) {
        return {
            {
                {0, 4, 6, 2}, // -X
                {1, 3, 7, 5}, // +X
                {0, 1, 5, 4}, // -Y
                {2, 6, 7, 3}, // +Y
                {0, 2, 3, 1}, // -Z (bottom)
                {4, 5, 7, 6} // +Z (top)
            }
        };
    }


    template<typename T>
    constexpr std::array<std::array<int, 3>, 12> GetFaceTris(const AABB<T> &) {
        return {
            {
                // -X
                {0, 4, 6}, {0, 6, 2},
                // +X
                {1, 3, 7}, {1, 7, 5},
                // -Y
                {0, 1, 5}, {0, 5, 4},
                // +Y
                {2, 6, 7}, {2, 7, 3},
                // -Z (bottom)
                {0, 2, 3}, {0, 3, 1},
                // +Z (top)
                {4, 5, 7}, {4, 7, 6}
            }
        };
    }

    template<typename T>
    constexpr std::array<Vector<int, 2>, 12> GetEdges(const AABB<T> &) {
        return {
            Vector<int, 2>{0, 1}, Vector<int, 2>{1, 3}, Vector<int, 2>{3, 2}, Vector<int, 2>{2, 0}, // Bottom face
            Vector<int, 2>{4, 5}, Vector<int, 2>{5, 7}, Vector<int, 2>{7, 6}, Vector<int, 2>{6, 4}, // Top face
            Vector<int, 2>{0, 4}, Vector<int, 2>{1, 5}, Vector<int, 2>{2, 6}, Vector<int, 2>{3, 7} // Vertical edges
        };
    }

    //BuilderTraits for AABB for Point, Vector of Points, OBB, Sphere
    template<typename T, typename Shape>
    struct BuilderTraits<AABB<T>, Shape> {
        CUDA_HOST_DEVICE static AABB<T> build(const Shape &) noexcept {
            static_assert(sizeof(Shape) == 0, "BuilderTraits<AABB, Shape> not implemented for this shape.");
            return AABB<T>();
        }

        CUDA_HOST_DEVICE static AABB<T> build(const std::vector<Vector<T, 3> > &shape) noexcept {
            return AABB<T>::Build(shape.begin(), shape.end());
        }

        CUDA_HOST_DEVICE static AABB<T> build(const OBB<T> &shape) noexcept {
            auto corners = GetVertices(shape);
            return AABB<T>::Build(corners.begin(), corners.end());
        }

        CUDA_HOST_DEVICE static AABB<T> build(const Sphere<T> &shape) noexcept {
            Vector<T, 3> min = shape.center - Vector<T, 3>(shape.radius);
            Vector<T, 3> max = shape.center + Vector<T, 3>(shape.radius);
            return AABB<T>{min, max};
        }

        CUDA_HOST static std::vector<AABB<T> > build_all(const std::vector<Vector<T, 3> > &points) noexcept {
            std::vector<AABB<T> > aabbs;
            for (const auto &p: points) {
                aabbs.emplace_back(p, p);
            }
            return aabbs;
        }
    };

    template<typename T>
    Vector<T, 3> ClosestPoint(const AABB<T> &aabb, const Vector<T, 3> &point) {
        return ClosestPointTraits<AABB<T>, Vector<T, 3> >::closest_point(aabb, point);
    }

    template<typename T>
    bool Contains(const AABB<T> &aabb, const Vector<T, 3> &point) {
        return ContainsTraits<AABB<T>, Vector<T, 3> >::contains(aabb, point);
    }

    template<typename T>
    bool Contains(const AABB<T> &aabb, const AABB<T> &other) {
        return ContainsTraits<AABB<T>, AABB<T> >::contains(aabb, other);
    }

    template<typename T>
    bool Intersects(const AABB<T> &a, const AABB<T> &b) {
        return IntersectsTraits<AABB<T>, AABB<T> >::intersects(a, b);
    }

    template<typename T>
    AABB<T> Intersection(const AABB<T> &a, const AABB<T> &b) {
        return IntersectionTraits<AABB<T>, AABB<T> >::intersects(a, b);
    }

    template<typename T>
    T Distance(const AABB<T> &aabb, const Vector<T, 3> &point) {
        return DistanceTraits<AABB<T>, AABB<T> >::distance(aabb, point);
    }

    template<typename T>
    AABB<T> apply_transform(const AABB<T> &aabb, const Matrix<T, 4, 4> &transform) {
        AABB<T> transformed_aabb;
        auto corners = GetVertices(aabb);
        for (auto iter = corners.begin(); iter != corners.end(); ++iter) {
            transformed_aabb.grow(glm::vec3(transform * glm::vec4(*iter, 1.0f)));
        }

        return transformed_aabb;
    }
}

#endif //AABBUTILS_H
