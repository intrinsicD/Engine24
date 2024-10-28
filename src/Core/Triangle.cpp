//
// Created by alex on 18.07.24.
//

#include "Triangle.h"

namespace Bcg{
    Vector<float, 3> VU(const Triangle &triangle) {
        return triangle.vertices.v - triangle.vertices.u;
    }


    Vector<float, 3> WU(const Triangle &triangle) {
        return triangle.vertices.w - triangle.vertices.u;
    }


    Vector<float, 3> WV(const Triangle &triangle) {
        return triangle.vertices.w - triangle.vertices.v;
    }


    float Area(const Triangle &triangle) {
        return Triangle::area(VU(triangle), WU(triangle));
    }

    Vector<float, 3> Normal(const Triangle &triangle) {
        return Triangle::normal(VU(triangle), WU(triangle));
    }


    Vector<float, 3> ToBarycentricCoordinates(const Triangle &triangle, const Vector<float, 3> &point) {
        return Triangle::to_barycentric_coordinates(VU(triangle), WU(triangle), point - triangle.vertices.u,
                                                    Normal(triangle));
    }


    Vector<float, 3> FromBarycentricCoordinates(const Triangle &triangle, const Vector<float, 3> &bc) {
        return Triangle::from_barycentric_coordinates(triangle.vertices.u, triangle.vertices.v, triangle.vertices.w, bc);
    }

    float Distance(const Triangle &triangle, const Vector<float, 3> &point) {
        Vector<float, 3> vu_ = VU(triangle);
        Vector<float, 3> wu_ = WU(triangle);
        Vector<float, 3> n = Triangle::normal(vu_, wu_);

        Vector<float, 3> bary = Triangle::to_barycentric_coordinates(vu_, wu_, point - triangle.vertices.u, n);
        bary = Triangle::clamped_barycentric_coordinates(bary);
        Vector<float, 3> closest = FromBarycentricCoordinates(triangle, bary);

        Vector<float, 3> diff = point - closest;
        float sign = glm::dot(n, diff);
        return sign > 0 ? glm::length(diff) : -glm::length(diff);
    }


    float UnsignedDistance(const Triangle &triangle, const Vector<float, 3> &point) {
        return std::abs(Distance(triangle, point));
    }


    Vector<float, 3> ClosestPoint(const Triangle &triangle, const Vector<float, 3> &point) {
        Vector<float, 3> bary = ToBarycentricCoordinates(triangle, point);
        bary = Triangle::clamped_barycentric_coordinates(bary);
        return FromBarycentricCoordinates(triangle, bary);
    }
}