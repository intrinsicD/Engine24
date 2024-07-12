//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_TRIANGLE_H
#define ENGINE24_TRIANGLE_H

#include "../MatVec.h"

namespace Bcg {
    template<typename T>
    struct Triangle {
        Vector<T, 3> u, v, w;

        static T area(const Vector<T, 3> vu, const Vector<T, 3> &wu) {
            return cross(vu, wu).norm();
        }

        static Vector<T, 3> to_barycentric_coordinates(const Vector<T, 3> &vu,
                                                       const Vector<T, 3> &wu,
                                                       const Vector<T, 3> &pu,
                                                       const Vector<T, 3> &normal) {
            T ax = std::abs(normal[0]), ay = std::abs(normal[1]), az = std::abs(normal[2]);
            Vector<T, 3> result;

            if (ax > ay && ax > az) {
                if (ax != 0) {
                    result[1] = (pu[1] * wu[2] - pu[2] * wu[1]) / normal[0];
                    result[2] = (vu[1] * pu[2] - vu[2] * pu[1]) / normal[0];
                    result[0] = 1 - result[1] - result[2];
                }
            } else if (ay > az) {
                if (ay != 0) {
                    result[1] = (pu[2] * wu[0] - pu[0] * wu[2]) / normal[1];
                    result[2] = (vu[2] * pu[0] - vu[0] * pu[2]) / normal[1];
                    result[0] = 1 - result[1] - result[2];
                }
            } else {
                if (az != 0) {
                    result[1] = (pu[0] * wu[1] - pu[1] * wu[0]) / normal[2];
                    result[2] = (vu[0] * pu[1] - vu[1] * pu[0]) / normal[2];
                    result[0] = 1 - result[1] - result[2];
                }
            }

            return result;
        }

        static Vector<T, 3> from_barycentric_coordinites(const Vector<T, 3> &u,
                                                         const Vector<T, 3> &v,
                                                         const Vector<T, 3> &w,
                                                         const Vector<T, 3> &bc) {
            return bc[0] * u + bc[1] * v + bc[2] * w;
        }

        static Vector<T, 3> normal(const Vector<T, 3> &vu, const Vector<T, 3> &wu) {
            return cross(vu, wu).normalized();
        }

        static Vector<T, 3> clamped_barycentric_coordinates(const Vector<T, 3> &bary) {
            Vector<T, 3> clamped = bary;
            clamped[1] = std::clamp(bary[1], T(0), T(1));
            clamped[2] = std::clamp(bary[2], T(0), T(1));
            clamped[0] = 1.0 - clamped[1] - clamped[2];

            if (clamped[0] < 0) {
                clamped[1] = std::clamp(clamped[1] / (clamped[1] + clamped[2]), T(0), T(1));
                clamped[2] = 1.0 - clamped[1];
            } else if (clamped[1] < 0) {
                clamped[2] = std::clamp(clamped[2] / (clamped[2] + clamped[0]), T(0), T(1));
                clamped[0] = 1.0 - clamped[2];
            } else if (clamped[2] < 0) {
                clamped[1] = std::clamp(clamped[1] / (clamped[1] + clamped[0]), T(0), T(1));
                clamped[0] = 1.0 - clamped[1];
            }
            return clamped;
        }
    };

    template<typename T>
    Vector<T, 3> vu(const Triangle<T> &triangle) {
        return triangle.v - triangle.u;
    }

    template<typename T>
    Vector<T, 3> wu(const Triangle<T> &triangle) {
        return triangle.w - triangle.u;
    }

    template<typename T>
    Vector<T, 3> wv(const Triangle<T> &triangle) {
        return triangle.w - triangle.v;
    }

    template<typename T>
    T area(const Triangle<T> &triangle) {
        return Triangle<T>::area(vu(triangle), wu(triangle));
    }

    template<typename T>
    Vector<T, 3> to_barycentric_coordinates(const Triangle<T> &triangle, const Vector<T, 3> &point) {
        return Triangle<T>::to_barycentric_coordinates(vu(triangle), wu(triangle), point - triangle.u,
                                                       Triangle<T>::normal(triangle));
    }

    template<typename T>
    Vector<T, 3> from_barycentric_coordinites(const Triangle<T> &triangle, const Vector<T, 3> &bc) {
        return Triangle<T>::from_barycentric_coordinites(triangle.u, triangle.v, triangle.w, bc);
    }

    template<typename T>
    Vector<T, 3> normal(const Triangle<T> &triangle) {
        return Triangle<T>::normal(vu(triangle), wu(triangle));
    }

    template<typename T>
    T distance(const Triangle<T> &triangle, const Vector<T, 3> &point) {
        Vector<T, 3> vu_ = vu(triangle);
        Vector<T, 3> wu_ = wu(triangle);
        Vector<T, 3> n = Triangle<T>::normal(vu_, wu_);

        Vector<T, 3> bary = Triangle<T>::to_barycentric_coordinates(vu_, wu_, point - triangle.u, n);
        bary = Triangle<T>::clamped_barycentric_coordinates(bary);
        Vector<T, 3> closest = from_barycentric_coordinates(triangle, bary);

        Vector<T, 3> diff = point - closest;
        T sign = n.dot(diff);
        return sign > 0 ? diff.norm() : -diff.norm();
    }

    template<typename T>
    T unsigned_distance(const Triangle<T> &triangle, const Vector<T, 3> &point) {
        return std::abs(Triangle<T>::distance(triangle, point));
    }

    template<typename T>
    Vector<T, 3> closest_point(const Triangle<T> &triangle, const Vector<T, 3> &point) {
        Vector<T, 3> bary = to_barycentric_coordinates(triangle, point);
        bary = Triangle<T>::clamped_barycentric_coordinates(bary);
        return from_barycentric_coordinates(triangle, bary);
    }
}

#endif //ENGINE24_TRIANGLE_H
