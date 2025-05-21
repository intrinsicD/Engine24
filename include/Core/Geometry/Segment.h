//
// Created by alex on 07.11.24.
//

#ifndef ENGINE24_SEGMENT_H
#define ENGINE24_SEGMENT_H

#include "MatVec.h"

namespace Bcg{
    template<typename T, int N>
    struct Segment {
        Eigen::Vector<T, N> start;
        Eigen::Vector<T, N> end;

        Eigen::Vector<T, N> diff() const  {
            return end - start;
        }

        Eigen::Vector<T, N> direction() const {
            return diff().normalized();
        }

        T length() const {
            return diff().norm();
        }
    };
}

#endif //ENGINE24_SEGMENT_H
