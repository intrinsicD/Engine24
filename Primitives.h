//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_PRIMITIVES_H
#define ENGINE24_PRIMITIVES_H

#include "Rotation.h"
#include "SurfaceMesh.h"

namespace Bcg {

    template<typename T>
    struct Plane {
        Vector<T, 3> normal;
        T d;

        T distance(const Vector<T, 3> &point) const {
            return normal.dot(point) + d;
        }

        T unsigned_distance(const Vector<T, 3> &point) const {
            return std::abs(distance(point));
        }
    };

    template<typename T>
    struct Frustum {
        union {
            Matrix<T, 6, 4> planes;
            Plane<T> plane[6];
        };
    };


    template<typename T>
    struct Sphere {
        Vector<T, 3> center;
        T radius;

        T distance(const Vector<T, 3> &point) const {
            return (point - center).norm() - radius;
        }

        T unsigned_distance(const Vector<T, 3> &point) const {
            return std::abs(distance(point));
        }
    };

    template<typename T>
    struct AABB {
        Vector<float, 3> min, max;

        T distance(const Vector<T, 3> &point) const {
            return (point - min.cwiseMax(point.cwiseMin(max))).norm();
        }

        T unsigned_distance(const Vector<T, 3> &point) const {
            return std::abs(distance(point));
        }
    };

    template<typename T>
    struct OBB {
        Vector<T, 3> center;
        Rotation<AngleAxis<T>> orientation;
        Vector<T, 3> half_extent;

        T distance(const Vector<T, 3> &point) const {
            Vector<T, 3> local_point = orientation.inverse() * (point - center);
            Vector<T, 3> closest_point;
            for (int i = 0; i < 3; ++i) {
                T dist = local_point[i];
                dist = std::max(dist, -half_extent[i]);
                dist = std::min(dist, half_extent[i]);
                closest_point[i] = dist;
            }
            return (local_point - closest_point).norm();
        }

        T unsigned_distance(const Vector<T, 3> &point) const {
            return std::abs(distance(point));
        }
    };

}

#endif //ENGINE24_PRIMITIVES_H
