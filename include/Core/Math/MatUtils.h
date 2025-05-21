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
}
#endif //ENGINE24_MATUTILS_H
