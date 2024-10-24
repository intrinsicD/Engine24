//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANKULLBACKLEIBLERDIVERGENCE_H
#define ENGINE24_GAUSSIANKULLBACKLEIBLERDIVERGENCE_H

#include "GaussianMahalonobisDistance.h"

namespace Bcg {
    template<typename T, int N>
    CUDA_HOST_DEVICE inline T KullbackLeiblerDiv(const Vector<T, N> &meanP, const Matrix<T, N, N> &covP,
                         const Vector<T, N> &meanQ, const Matrix<T, N, N> &covQ) {
        Matrix<T, N, N> invCovQ = covQ.inverse();
        T tr = (invCovQ * covP).trace();
        T logDet = std::log(covQ.determinant() / covP.determinant());
        return 0.5 * (tr + SquaredMahalonobisDistance(meanP, meanQ, invCovQ) - N + logDet);
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline T KullbackLeiblerDiv(const GaussianStruct<T, N> &P, const GaussianStruct<T, N> &Q) {
        return KullbackLeiblerDiv(P.mean, P.covariance, Q.mean, Q.covariance);
    }
}

#endif //ENGINE24_GAUSSIANKULLBACKLEIBLERDIVERGENCE_H
