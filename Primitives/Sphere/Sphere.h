//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_SPHERE_H
#define ENGINE24_SPHERE_H

#include "../MatVec.h"

namespace Bcg{
    template<typename T>
    struct Sphere {
        Vector<T, 3> center;
        T radius;
    };

    template<typename T>
    Vector<T, 3> closest_point(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return sphere.center + (point - sphere.center).normalized() * sphere.radius;
    }

    template<typename T>
    T volume(const Sphere<T> &sphere) {
        return (4.0 / 3.0) * M_PI * std::pow(sphere.radius, 3);
    }

    template<typename T>
    T surface_area(const Sphere<T> &sphere) {
        return 4.0 * M_PI * std::pow(sphere.radius, 2);
    }

    template<typename T>
    T distance(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return (point - sphere.center).norm() - sphere.radius;
    }

    template<typename T>
    T unsigned_distance(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return std::abs(Sphere<T>::distance(sphere, point));
    }
}
#endif //ENGINE24_SPHERE_H
