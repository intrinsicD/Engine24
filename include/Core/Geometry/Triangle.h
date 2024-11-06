//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_TRIANGLE_H
#define ENGINE24_TRIANGLE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    struct TriangleBase {
        union{
            Vector<T, 3> u, v, w;
            Matrix<T, 3, 3> matrix;
            Vector<T, 3> array[3];
            std::vector<Vector<T, 3>> vector;
        }vertices;


        static T area(const Vector<T, 3> vu, const Vector<T, 3> &wu) {
            return glm::length(cross(vu, wu));
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

        static Vector<T, 3> from_barycentric_coordinates(const Vector<T, 3> &u,
                                                         const Vector<T, 3> &v,
                                                         const Vector<T, 3> &w,
                                                         const Vector<T, 3> &bc) {
            return bc[0] * u + bc[1] * v + bc[2] * w;
        }

        static Vector<T, 3> normal(const Vector<T, 3> &vu, const Vector<T, 3> &wu) {
            return glm::normalize(cross(vu, wu));
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

        std::vector<Vector<float, 3>> Edges() const {
            return {vertices.v - vertices.u, vertices.w - vertices.v, vertices.u - vertices.w};
        }
    };

    using Trianglef = TriangleBase<float>;
    using Triangle = Trianglef;

    Vector<float, 3> VU(const Triangle &triangle);

    Vector<float, 3> WU(const Triangle &triangle) ;

    Vector<float, 3> WV(const Triangle &triangle);

    float Area(const Triangle &triangle);

    Vector<float, 3> Normal(const Triangle &triangle);

    Vector<float, 3> ToBarycentricCoordinates(const Triangle &triangle, const Vector<float, 3> &point) ;

    Vector<float, 3> FromBarycentricCoordinates(const Triangle &triangle, const Vector<float, 3> &bc) ;

    float Distance(const Triangle &triangle, const Vector<float, 3> &point);

    float UnsignedDistance(const Triangle &triangle, const Vector<float, 3> &point);

    Vector<float, 3> ClosestPoint(const Triangle &triangle, const Vector<float, 3> &point);
}

#endif //ENGINE24_TRIANGLE_H
