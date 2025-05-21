//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_LINEUTILS_H
#define ENGINE24_LINEUTILS_H

#include "Line.h"

namespace Bcg{
    template<typename T, int N>
    Eigen::Vector<T, N> ClosestPoint(const Line<T, N> &line, const Eigen::Vector<T, N> &point) {
        return line.base + ((point - line.base).dot(line.direction) / line.direction.squaredNorm()) * line.direction;
    }

    template<typename T, int N>
    T Distance(const Line<T, N> &line, const Eigen::Vector<T, N> &point) {
        return (point - ClosestPoint(line, point)).norm();
    }
}

#endif //ENGINE24_LINEUTILS_H
