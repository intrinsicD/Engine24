#pragma once

#include "MatVec.h"
#include "Macros.h"
#include "GeometricTraits.h"

namespace Bcg {
    template <typename T>
    struct OBB {
        Vector<T, 3> center;
        Vector<T, 3> half_extents; // Half the lengths along each principal axis
        Quaternion<T> axes; // Each column is a principal axis (should be orthonormal)
    };

    // Closest point on OBB to a point
    template<typename T>
    struct ClosestPointTraits<OBB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static Vector<T, 3> closest_point(const OBB<T> &obb, const Vector<T, 3> &point) noexcept {
            // Orientation matrix whose columns are the OBB axes in world space
            auto R = glm::mat3_cast(obb.axes);
            const Vector<T, 3> a0 = R[0];
            const Vector<T, 3> a1 = R[1];
            const Vector<T, 3> a2 = R[2];

            // Vector from center to point
            const Vector<T, 3> d = point - obb.center;

            // Project onto axes (local coordinates)
            const T x = glm::dot(d, a0);
            const T y = glm::dot(d, a1);
            const T z = glm::dot(d, a2);

            // Clamp to half extents
            const Vector<T, 3> he = obb.half_extents;
            const T cx = glm::clamp(x, -he.x, he.x);
            const T cy = glm::clamp(y, -he.y, he.y);
            const T cz = glm::clamp(z, -he.z, he.z);

            // Reconstruct closest point in world space
            return obb.center + a0 * cx + a1 * cy + a2 * cz;
        }
    };

    template<typename T>
    struct SquaredDistanceTraits<OBB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static T squared_distance(const OBB<T> &obb, const Vector<T, 3> &point) noexcept {
            const Vector<T, 3> cp = ClosestPointTraits<OBB<T>, Vector<T, 3> >::closest_point(obb, point);
            const Vector<T, 3> diff = point - cp;
            return glm::dot(diff, diff);
        }
    };

    template<typename T>
    struct DistanceTraits<OBB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static T distance(const OBB<T> &obb, const Vector<T, 3> &point) noexcept {
            return sqrt(SquaredDistanceTraits<OBB<T>, Vector<T, 3> >::squared_distance(obb, point));
        }
    };

    // Point containment
    template<typename T>
    struct ContainsTraits<OBB<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool contains(const OBB<T> &obb, const Vector<T, 3> &point) noexcept {
            auto R = glm::mat3_cast(obb.axes);
            const Vector<T, 3> a0 = R[0];
            const Vector<T, 3> a1 = R[1];
            const Vector<T, 3> a2 = R[2];
            const Vector<T, 3> d = point - obb.center;

            const T x = glm::dot(d, a0);
            const T y = glm::dot(d, a1);
            const T z = glm::dot(d, a2);

            const Vector<T, 3> he = obb.half_extents;
            const T eps = T(1e-6);
            return (std::abs(x) <= he.x + eps) &&
                   (std::abs(y) <= he.y + eps) &&
                   (std::abs(z) <= he.z + eps);
        }
    };

    // OBB containment (sufficient test: all 8 vertices of inner are inside outer)
    template<typename T>
    struct ContainsTraits<OBB<T>, OBB<T> > {
        CUDA_HOST_DEVICE static bool contains(const OBB<T> &outer, const OBB<T> &inner) noexcept {
            // Build inner vertices in world space
            auto Rb = glm::mat3_cast(inner.axes);
            const Vector<T, 3> b0 = Rb[0];
            const Vector<T, 3> b1 = Rb[1];
            const Vector<T, 3> b2 = Rb[2];
            const Vector<T, 3> he = inner.half_extents;

            Vector<T, 3> verts[8];
            int idx = 0;
            for (int sx = -1; sx <= 1; sx += 2) {
                for (int sy = -1; sy <= 1; sy += 2) {
                    for (int sz = -1; sz <= 1; sz += 2) {
                        const Vector<T, 3> offset = b0 * (T(sx) * he.x) + b1 * (T(sy) * he.y) + b2 * (T(sz) * he.z);
                        verts[idx++] = inner.center + offset;
                    }
                }
            }

            for (const auto &v : verts) {
                if (!ContainsTraits<OBB<T>, Vector<T, 3> >::contains(outer, v)) return false;
            }
            return true;
        }
    };

    // OBB-OBB intersection using the Separating Axis Theorem (15 axes)
    template<typename T>
    struct IntersectsTraits<OBB<T>, OBB<T> > {
        CUDA_HOST_DEVICE static bool intersects(const OBB<T> &A, const OBB<T> &B) noexcept {
            // Compute rotation matrices (columns are axis directions)
            const auto RA = glm::mat3_cast(A.axes);
            const auto RB = glm::mat3_cast(B.axes);

            // Convenience axis vectors for A and B
            const Vector<T, 3> A0 = RA[0], A1 = RA[1], A2 = RA[2];
            const Vector<T, 3> B0 = RB[0], B1 = RB[1], B2 = RB[2];

            // Rotation matrix expressing B in A's frame: R[i][j] = Ai dot Bj
            T R[3][3];
            R[0][0] = glm::dot(A0, B0); R[0][1] = glm::dot(A0, B1); R[0][2] = glm::dot(A0, B2);
            R[1][0] = glm::dot(A1, B0); R[1][1] = glm::dot(A1, B1); R[1][2] = glm::dot(A1, B2);
            R[2][0] = glm::dot(A2, B0); R[2][1] = glm::dot(A2, B1); R[2][2] = glm::dot(A2, B2);

            // Compute translation t in A's frame
            const Vector<T, 3> tWorld = B.center - A.center;
            const T tA[3] = { glm::dot(tWorld, A0), glm::dot(tWorld, A1), glm::dot(tWorld, A2) };

            // Compute common subexpressions. Add in an epsilon term to
            // counteract arithmetic errors when two edges are parallel.
            const T EPS = T(1e-6);
            T AbsR[3][3];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    AbsR[i][j] = std::abs(R[i][j]) + EPS;
                }
            }

            const Vector<T, 3> a = A.half_extents;
            const Vector<T, 3> b = B.half_extents;

            T ra, rb;

            // Test axes L = A0, A1, A2
            for (int i = 0; i < 3; ++i) {
                ra = (&a.x)[i];
                rb = b.x * AbsR[i][0] + b.y * AbsR[i][1] + b.z * AbsR[i][2];
                if (std::abs(tA[i]) > ra + rb) return false;
            }

            // Test axes L = B0, B1, B2
            for (int j = 0; j < 3; ++j) {
                ra = a.x * AbsR[0][j] + a.y * AbsR[1][j] + a.z * AbsR[2][j];
                rb = (&b.x)[j];
                const T tB = tA[0] * R[0][j] + tA[1] * R[1][j] + tA[2] * R[2][j];
                if (std::abs(tB) > ra + rb) return false;
            }

            // Test axis L = Ai x Bj
            // i = 0
            ra = a.y * AbsR[2][0] + a.z * AbsR[1][0];
            rb = b.y * AbsR[0][2] + b.z * AbsR[0][1];
            if (std::abs(tA[2] * R[1][0] - tA[1] * R[2][0]) > ra + rb) return false;

            ra = a.y * AbsR[2][1] + a.z * AbsR[1][1];
            rb = b.x * AbsR[0][2] + b.z * AbsR[0][0];
            if (std::abs(tA[2] * R[1][1] - tA[1] * R[2][1]) > ra + rb) return false;

            ra = a.y * AbsR[2][2] + a.z * AbsR[1][2];
            rb = b.x * AbsR[0][1] + b.y * AbsR[0][0];
            if (std::abs(tA[2] * R[1][2] - tA[1] * R[2][2]) > ra + rb) return false;

            // i = 1
            ra = a.x * AbsR[2][0] + a.z * AbsR[0][0];
            rb = b.y * AbsR[1][2] + b.z * AbsR[1][1];
            if (std::abs(tA[0] * R[2][0] - tA[2] * R[0][0]) > ra + rb) return false;

            ra = a.x * AbsR[2][1] + a.z * AbsR[0][1];
            rb = b.x * AbsR[1][2] + b.z * AbsR[1][0];
            if (std::abs(tA[0] * R[2][1] - tA[2] * R[0][1]) > ra + rb) return false;

            ra = a.x * AbsR[2][2] + a.z * AbsR[0][2];
            rb = b.x * AbsR[1][1] + b.y * AbsR[1][0];
            if (std::abs(tA[0] * R[2][2] - tA[2] * R[0][2]) > ra + rb) return false;

            // i = 2
            ra = a.x * AbsR[1][0] + a.y * AbsR[0][0];
            rb = b.y * AbsR[2][2] + b.z * AbsR[2][1];
            if (std::abs(tA[1] * R[0][0] - tA[0] * R[1][0]) > ra + rb) return false;

            ra = a.x * AbsR[1][1] + a.y * AbsR[0][1];
            rb = b.x * AbsR[2][2] + b.z * AbsR[2][0];
            if (std::abs(tA[1] * R[0][1] - tA[0] * R[1][1]) > ra + rb) return false;

            ra = a.x * AbsR[1][2] + a.y * AbsR[0][2];
            rb = b.x * AbsR[2][1] + b.y * AbsR[2][0];
            if (std::abs(tA[1] * R[0][2] - tA[0] * R[1][2]) > ra + rb) return false;

            // No separating axis found
            return true;
        }
    };
}
