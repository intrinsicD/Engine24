//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANSTRUCT_H
#define ENGINE24_GAUSSIANSTRUCT_H

#include "MatVec.h"
#include "Macros.h"

namespace Bcg {
    template<typename T, int N>
    struct GaussianStruct {
        Vector<T, N> mean = Vector<T, N>::Zero();
        Matrix<T, N, N> covariance = Matrix<T, N, N>::Identity();
        T weight = 1.0;

        CUDA_HOST_DEVICE GaussianStruct(const Vector<T, N> &mean, const Matrix<T, N, N> &covariance, T weight = 1.0)
                : mean(mean), covariance(covariance), weight(weight) {}
    };
}

#endif //ENGINE24_GAUSSIANSTRUCT_H
