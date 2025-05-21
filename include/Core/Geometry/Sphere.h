//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_SPHERE_H
#define ENGINE24_SPHERE_H

#include "MatVec.h"
#include "StringTraits.h"

namespace Bcg {
    template<typename T, int N>
    struct Sphere {
        Eigen::Vector<T, N> center;
        T radius;

        Sphere() : center(0), radius(0) {}

        Sphere(const Eigen::Vector<T, N> &center, T radius) : center(center), radius(radius) {}

        T volume() const {
            // Calculate the volume of the nd sphere
            return (1.0 / 2) * M_PI * std::pow(radius, N);
        }

        T surface_area() const {
            // Calculate the surface area of the nd sphere
            return N * M_PI * std::pow(radius, N - 1);
        }
    };
}
#endif //ENGINE24_SPHERE_H
