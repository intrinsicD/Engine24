//
// Created by alex on 11/17/24.
//

#ifndef AABBUTILS_H
#define AABBUTILS_H

#include "AABB.h"

namespace Bcg::AABBUtils {
    /**
      * @brief Finds the closest point on the AABB to a given point.
      * @param aabb The Axis-Aligned Bounding Box.
      * @param point The point to find the closest point to.
      * @return The point on the AABB closest to the given point.
      */
    template<typename T, int N>
    inline Eigen::Vector<T, N> ClosestPoint(const AABB<T, N> &aabb, const Eigen::Vector<T, N> &point) {
        // For each dimension, clamp the point's coordinate to the AABB's range for that dimension.
        // result_i = std::clamp(point_i, aabb.min_i, aabb.max_i)
        // Using Eigen's cwise operations:
        return point.cwiseMax(aabb.min).cwiseMin(aabb.max);
    }

    /**
     * @brief Checks if a point is contained within an AABB.
     * A point on the boundary is considered contained.
     * @param aabb The Axis-Aligned Bounding Box.
     * @param point The point to check.
     * @return True if the point is contained within the AABB, false otherwise.
     */
    template<typename T, int N>
    inline bool Contains(const AABB<T, N> &aabb, const Eigen::Vector<T, N> &point) {
        // Check if all point coordinates are within the AABB's min/max bounds.
        // aabb.min_i <= point_i <= aabb.max_i for all i
        return (point.array() >= aabb.min.array()).all() &&
               (point.array() <= aabb.max.array()).all();
    }

    /**
     * @brief Checks if an AABB ('aabb') completely contains another AABB ('other').
     * If 'other' is touching the boundary of 'aabb' from the inside, it's considered contained.
     * @param aabb The containing Axis-Aligned Bounding Box.
     * @param other The Axis-Aligned Bounding Box to check for containment.
     * @return True if 'aabb' contains 'other', false otherwise.
     */
    template<typename T, int N>
    inline bool Contains(const AABB<T, N> &aabb, const AABB<T, N> &other) {
        // Check if 'aabb' encloses 'other'.
        // aabb.min_i <= other.min_i AND other.max_i <= aabb.max_i for all i
        return (aabb.min.array() <= other.min.array()).all() &&
               (other.max.array() <= aabb.max.array()).all();
    }

    /**
     * @brief Checks if two AABBs intersect.
     * AABBs that are merely touching are considered intersecting.
     * @param a The first Axis-Aligned Bounding Box.
     * @param b The second Axis-Aligned Bounding Box.
     * @return True if the AABBs intersect, false otherwise.
     */
    template<typename T, int N>
    inline bool Intersects(const AABB<T, N> &a, const AABB<T, N> &b) {
        // Two AABBs intersect if they overlap on all axes.
        // They don't intersect if there's at least one axis of separation.
        // No separation on axis i if: a.max_i >= b.min_i AND b.max_i >= a.min_i
        return (a.max.array() >= b.min.array()).all() &&
               (b.max.array() >= a.min.array()).all();
    }

    /**
     * @brief Computes the intersection of two AABBs.
     * @param a The first Axis-Aligned Bounding Box.
     * @param b The second Axis-Aligned Bounding Box.
     * @return A new AABB representing the intersection. If they don't intersect,
     *         the resulting AABB may be "invalid" (e.g., min_i > max_i for some i).
     *         You can check this with a helper like `result_aabb.isValid()`.
     */
    template<typename T, int N>
    inline AABB<T, N> Intersection(const AABB<T, N> &a, const AABB<T, N> &b) {
        AABB<T, N> intersection_aabb;
        intersection_aabb.min = a.min.cwiseMax(b.min);
        intersection_aabb.max = a.max.cwiseMin(b.max);
        return intersection_aabb;
    }

    /**
     * @brief Calculates the shortest distance from a point to an AABB.
     * If the point is inside the AABB, the distance is 0.
     * @param aabb The Axis-Aligned Bounding Box.
     * @param point The point.
     * @return The Euclidean distance as a float.
     */
    template<typename T, int N>
    inline T Distance(const AABB<T, N> &aabb, const Eigen::Vector<T, N> &point) {
        Eigen::Vector<T, N> closestPt = ClosestPoint(aabb, point);
        // Eigen's norm() method computes the L2 norm (Euclidean distance).
        // It returns RealScalar, which is T if T is float/double, or double if T is int.
        // This handles potential promotion for integer types correctly.
        return (point - closestPt).norm();
    }
}

namespace Bcg {
    /**
     * @brief String representation of the AABB.
     * @param aabb The Axis-Aligned Bounding Box.
     * @return A string representation of the AABB.
     */
    template<typename S, int N>
    struct StringTraits<AABB<S, N>> {
        static std::string ToString(const AABB<S, N> &aabb) {
            std::stringstream ss;
            ss << "AABB(min=" << aabb.min.transpose() << ", max=" << aabb.max.transpose() << ")";
            return ss.str();
        }
    };
}

#endif //AABBUTILS_H
