//
// Created by alex on 25.10.24.
//

#ifndef ENGINE24_MATUTILS_H
#define ENGINE24_MATUTILS_H

#include "MatVec.h"

namespace Bcg{
    template<typename T, int N>
    inline T Trace(const Eigen::Matrix<T, N, N> &m) {
        return m.diagonal().sum();
    }

    template<typename T>
    T RadiansToDegrees(T radians) {
        return radians * (180.0f / M_PI);
    }

    template<typename T>
    T DegreesToRadians(T degrees) {
        return degrees * (M_PI / 180.0f);
    }
}
#endif //ENGINE24_MATUTILS_H
