//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATVEC_H
#define ENGINE24_MATVEC_H

#include "Eigen/Core"

namespace Bcg {
    using Scalar = float;

    template<typename T, int N>
    using Vector = Eigen::Vector<T, N>;

    template<typename T, int M, int N>
    using Matrix = Eigen::Matrix<T, M, N>;
}

#endif //ENGINE24_MATVEC_H
