//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_TRIANGLE_H
#define ENGINE24_TRIANGLE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T, int N>
    struct Triangle {
        Eigen::Vector<T, N> u, v, w;

        Triangle() = default;

        Triangle(const Eigen::Vector<T, N> &u_, const Eigen::Vector<T, N> &v_, const Eigen::Vector<T, N> &w_) {
            u = u_;
            v = v_;
            w = w_;
        }

        T area() const {
            return (v - u).cross(w - u).norm() / 2.0;
        }

        T perimeter() const {
            return (u - v).norm() + (v - w).norm() + (w - u).norm();
        }

    };
}

#endif //ENGINE24_TRIANGLE_H
