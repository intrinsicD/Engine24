//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_SEGMENTUTILS_H
#define ENGINE24_SEGMENTUTILS_H

#include "Segment.h"

namespace Bcg{
    template<typename T, int N>
    Eigen::Vector<T, N> ClosestPoint(const Segment<T, N> &segment, const Eigen::Vector<T, N> &point) {
        return segment.start + segment.direction() * (point - segment.start).dot(segment.direction());
    }

    template<typename T, int N>
    T Distance(const Segment<T, N> &segment, const Eigen::Vector<T, N> &point) {
        return (point - ClosestPoint(segment, point)).norm();
    }
}

#endif //ENGINE24_SEGMENTUTILS_H
