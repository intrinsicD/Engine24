//
// Created by alex on 18.07.24.
//

#include "Sphere.h"

namespace Bcg{
    Vector<float, 3> ClosestPoint(const Sphere &sphere, const Vector<float, 3> &point) {
        return sphere.center + glm::normalize(point - sphere.center) * sphere.radius;
    }


    float Volume(const Sphere &sphere) {
        return (4.0f / 3.0f) * M_PI * std::pow(sphere.radius, 3);
    }


    float SurfaceArea(const Sphere &sphere) {
        return 4.0f * M_PI * std::pow(sphere.radius, 2);
    }


    float Distance(const Sphere &sphere, const Vector<float, 3> &point) {
        return glm::length(point - sphere.center) - sphere.radius;
    }


    float UnsignedDistance(const Sphere &sphere, const Vector<float, 3> &point) {
        return std::abs(Distance(sphere, point));
    }
}