//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_LINE_H
#define ENGINE24_LINE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T, int N>
    struct Line {
        Eigen::Vector<T, N> base;
        Eigen::Vector<T, N> direction;

        Line(const Eigen::Vector<T, N> &base, const Eigen::Vector<T, N> &direction) : base(base), direction(direction) {}
    };

}

#endif //ENGINE24_LINE_H
