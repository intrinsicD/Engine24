//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_SPHEREUTILS_H
#define ENGINE24_SPHEREUTILS_H

#include "Sphere.h"

namespace Bcg {
    template<typename T, int N>
    Eigen::Vector<T, N> ClosestPoint(const Sphere<T, N> &sphere, const Eigen::Vector<T, N> &point) {
        return sphere.center + (point - sphere.center).normalized() * sphere.radius;
    }

    template<typename T, int N>
    T Distance(const Sphere<T, N> &sphere, const Eigen::Vector<T, N> &point) {
        return (point - sphere.center).norm() - sphere.radius;
    }
}

#endif //ENGINE24_SPHEREUTILS_H
