//
// Created by alex on 18.07.24.
//

#include "Sphere.h"

namespace Bcg{
    Vector<float, 3> ClosestPoint(const Sphere &sphere, const Vector<float, 3> &point) {
        return sphere.center + (point - sphere.center).normalized() * sphere.radius;
    }


    float Volume(const Sphere &sphere) {
        return (4.0 / 3.0) * M_PI * std::pow(sphere.radius, 3);
    }


    float SurfaceArea(const Sphere &sphere) {
        return 4.0 * M_PI * std::pow(sphere.radius, 2);
    }


    float Distance(const Sphere &sphere, const Vector<float, 3> &point) {
        return (point - sphere.center).norm() - sphere.radius;
    }


    float UnsignedDistance(const Sphere &sphere, const Vector<float, 3> &point) {
        return std::abs(Distance(sphere, point));
    }
}