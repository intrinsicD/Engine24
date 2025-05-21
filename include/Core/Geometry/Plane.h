//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_PLANE_H
#define ENGINE24_PLANE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T, int N>
    struct Plane {
        Eigen::Vector<T, N> normal;
        T d;

        Plane(const Eigen::Vector<T, N> &normal, T d) : normal(normal), d(d) {}

        Plane(const Eigen::Vector<T, N> &normal, const Eigen::Vector<T, N> &point) : normal(normal), d(-normal.dot(point)) {}
    };
}

#endif //ENGINE24_PLANE_H
