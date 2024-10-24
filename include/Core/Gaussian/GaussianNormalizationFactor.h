//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANNORMALIZATIONFACTOR_H
#define ENGINE24_GAUSSIANNORMALIZATIONFACTOR_H

#include "GaussianStruct.h"

namespace Bcg{
    template<typename T, int N>
    CUDA_HOST_DEVICE inline T GaussianNormalizationFactor(const Matrix<T, N, N> &cov) {
        return std::sqrt(std::pow(2 * M_PI, N) * cov.determinant());
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline T GaussianNormalizationFactor(const GaussianStruct<T, N> &P) {
        return GaussianNormalizationFactor(P.covariance);
    }
}


#endif //ENGINE24_GAUSSIANNORMALIZATIONFACTOR_H
