#pragma once

#include "OBB.h"
#include "AABB.h"
#include "Sphere.h"
#include "GeometricTraits.h"
#include "GlmToEigen.h"
#include <Eigen/Eigenvalues>

namespace Bcg{
    template<typename T>
    std::array<Vector<T, 3>, 8> GetVertices(const OBB<T> &obb) {
        std::array<Vector<T, 3>, 8> vertices{};
        Matrix<T, 3, 3> R = glm::mat3_cast(obb.axes);
        Vector<T, 3> he = obb.half_extents;

        // Calculate the 8 vertices of the OBB
        vertices[0] = obb.center + R * Vector<T, 3>(-he.x, -he.y, -he.z);
        vertices[1] = obb.center + R * Vector<T, 3>(he.x, -he.y, -he.z);
        vertices[2] = obb.center + R * Vector<T, 3>(he.x, he.y, -he.z);
        vertices[3] = obb.center + R * Vector<T, 3>(-he.x, he.y, -he.z);
        vertices[4] = obb.center + R * Vector<T, 3>(-he.x, -he.y, he.z);
        vertices[5] = obb.center + R * Vector<T, 3>(he.x, -he.y, he.z);
        vertices[6] = obb.center + R * Vector<T, 3>(he.x, he.y, he.z);
        vertices[7] = obb.center + R * Vector<T, 3>(-he.x, he.y, he.z);

        return vertices;
    }

    template<typename T>
    constexpr std::array<Vector<int, 2>, 12> GetEdges(const OBB<T> &) {
        return {
            Vector<int, 2>{0, 1}, Vector<int, 2>{1, 2}, Vector<int, 2>{2, 3}, Vector<int, 2>{3, 0}, // Bottom face
            Vector<int, 2>{4, 5}, Vector<int, 2>{5, 6}, Vector<int, 2>{6, 7}, Vector<int, 2>{7, 4}, // Top face
            Vector<int, 2>{0, 4}, Vector<int, 2>{1, 5}, Vector<int, 2>{2, 6}, Vector<int, 2>{3, 7} // Side edges
        };
    }

    template<typename T>
    constexpr std::array<std::array<int, 3>, 12> GetFaceTris(const OBB<T> &) {
        return {
            std::array<int, 3>{0, 1, 2}, std::array<int, 3>{0, 2, 3}, // Bottom face
            std::array<int, 3>{4, 5, 6}, std::array<int, 3>{4, 6, 7}, // Top face
            std::array<int, 3>{0, 1, 5}, std::array<int, 3>{0, 5, 4}, // Side face
            std::array<int, 3>{1, 2, 6}, std::array<int, 3>{1, 6, 5}, // Side face
            std::array<int, 3>{2, 3, 7}, std::array<int, 3>{2, 7, 6}, // Side face
            std::array<int, 3>{3, 0, 4}, std::array<int, 3>{3, 4, 7} // Side face
        };
    }

    //BuilderTraits of OBB for Point, AABB, Sphere
    template<typename T, typename Shape>
    struct BuilderTraits<OBB<T>, Shape> {
        CUDA_HOST_DEVICE static OBB<T> build(const Shape &) noexcept {
            static_assert(sizeof(Shape) == 0, "BuilderTraits<OBB, Shape> not implemented for this shape.");
            return OBB<T>();
        }

        CUDA_HOST_DEVICE static OBB<T> build(const AABB<T> &shape) noexcept {
            Vector<T, 3> center = (shape.min + shape.max) * static_cast<T>(0.5);
            Vector<T, 3> half_extents = (shape.max - shape.min) * static_cast<T>(0.5);
            Vector<T, 4> axes(1, 0, 0, 0); // Identity quaternion
            return OBB<T>{center, half_extents, axes};
        }

        CUDA_HOST_DEVICE static OBB<T> build(const Sphere<T> &shape) noexcept {
            Vector<T, 3> center = shape.center;
            Vector<T, 3> half_extents(shape.radius, shape.radius, shape.radius);
            Vector<T, 4> axes(1, 0, 0, 0); // Identity quaternion
            return OBB<T>{center, half_extents, axes};
        }

        CUDA_HOST_DEVICE static OBB<T> build(const Vector<T, 3> &shape) noexcept {
            return OBB<T>{shape, Vector<T, 3>(0, 0, 0), Vector<T, 4>(1, 0, 0, 0)};
        }

        CUDA_HOST_DEVICE static OBB<T> build(const Vector<T, 3> &center, const Vector<T, 3> &scale, const Quaternion<T> &quat) noexcept {
            return OBB<T>{center, scale, quat};
        }

        CUDA_HOST static std::vector<OBB<T>> build_all(const std::vector<Vector<T, 3> > &points) noexcept {
            std::vector<OBB<T>> obbs;
            for (const auto &p: points) {
                obbs.emplace_back(p, Vector<T, 3>(0, 0, 0), Vector<T, 4>(1, 0, 0, 0));
            }
            return obbs;
        }
    };

    // Builder for OBB from a vector of points: axes from PCA, extents from projections
    template<typename T>
    struct BuilderTraits<OBB<T>, std::vector<Vector<T, 3>>> {
        CUDA_HOST static OBB<T> build(const std::vector<Vector<T, 3>> &points) noexcept {
            OBB<T> result{};
            if (points.empty()) {
                result.center = Vector<T,3>(0);
                result.half_extents = Vector<T,3>(0);
                result.axes = Vector<T,4>(1,0,0,0);
                return result;
            }

            // 1) Compute centroid
            Vector<T,3> c(0);
            for (const auto &p : points) c += p;
            c /= static_cast<T>(points.size());

            // 2) Compute covariance matrix
            Eigen::Matrix<T,3,3> C = Eigen::Matrix<T,3,3>::Zero();
            for (const auto &p : points) {
                Eigen::Matrix<T,3,1> d;
                d << (p.x - c.x), (p.y - c.y), (p.z - c.z);
                C.noalias() += d * d.transpose();
            }
            C /= static_cast<T>(points.size());

            // 3) Eigen decomposition (covariance is symmetric)
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,3,3>> es(C);
            const auto &V = es.eigenvectors(); // columns are eigenvectors (ascending eigenvalues)

            // Build rotation matrix with columns = eigenvectors
            Matrix<T,3,3> R(1);
            R[0] = Vector<T,3>(V(0,0), V(1,0), V(2,0));
            R[1] = Vector<T,3>(V(0,1), V(1,1), V(2,1));
            R[2] = Vector<T,3>(V(0,2), V(1,2), V(2,2));

            // Ensure right-handed basis
            if (glm::dot(R[2], glm::cross(R[0], R[1])) < T(0)) {
                R[2] = -R[2];
            }

            // 4) Project points into local frame to find min/max on each axis
            Vector<T,3> minL(std::numeric_limits<T>::max());
            Vector<T,3> maxL(std::numeric_limits<T>::lowest());
            for (const auto &p : points) {
                const Vector<T,3> d = p - c;
                const T lx = glm::dot(d, R[0]);
                const T ly = glm::dot(d, R[1]);
                const T lz = glm::dot(d, R[2]);
                minL.x = glm::min(minL.x, lx); maxL.x = glm::max(maxL.x, lx);
                minL.y = glm::min(minL.y, ly); maxL.y = glm::max(maxL.y, ly);
                minL.z = glm::min(minL.z, lz); maxL.z = glm::max(maxL.z, lz);
            }

            const Vector<T,3> he = (maxL - minL) * T(0.5);
            const Vector<T,3> centerLocal = (minL + maxL) * T(0.5);
            const Vector<T,3> centerWorld = c + R * centerLocal;

            result.center = centerWorld;
            result.half_extents = he;
            result.axes = glm::quat_cast(R);
            return result;
        }
    };

    // Convenience wrappers using traits
    template<typename T>
    Vector<T, 3> ClosestPoint(const OBB<T> &obb, const Vector<T, 3> &point) {
        return ClosestPointTraits<OBB<T>, Vector<T, 3> >::closest_point(obb, point);
    }

    template<typename T>
    T SquaredDistance(const OBB<T> &obb, const Vector<T, 3> &point) {
        return SquaredDistanceTraits<OBB<T>, Vector<T, 3> >::squared_distance(obb, point);
    }

    template<typename T>
    T Distance(const OBB<T> &obb, const Vector<T, 3> &point) {
        return DistanceTraits<OBB<T>, Vector<T, 3> >::distance(obb, point);
    }

    template<typename T>
    bool Contains(const OBB<T> &obb, const Vector<T, 3> &point) {
        return ContainsTraits<OBB<T>, Vector<T, 3> >::contains(obb, point);
    }

    template<typename T>
    bool Contains(const OBB<T> &outer, const OBB<T> &inner) {
        return ContainsTraits<OBB<T>, OBB<T> >::contains(outer, inner);
    }

    template<typename T>
    bool Intersects(const OBB<T> &a, const OBB<T> &b) {
        return IntersectsTraits<OBB<T>, OBB<T> >::intersects(a, b);
    }
}
