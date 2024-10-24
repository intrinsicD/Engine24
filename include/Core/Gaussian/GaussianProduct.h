//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANPRODUCT_H
#define ENGINE24_GAUSSIANPRODUCT_H

#include "GaussianMahalonobisDistance.h"
#include "GaussianNormalizationFactor.h"

namespace Bcg {
    template<typename T, int N>
    CUDA_HOST_DEVICE inline GaussianStruct<T, N>
    MultiplyGaussians(const Vector<T, N> &meanP, const Matrix<T, N, N> &covP, T weightP,
                      const Vector<T, N> &meanQ, const Matrix<T, N, N> &covQ, T weightQ) {
        Matrix<T, N, N> covSum = covP + covQ;
        Matrix<T, N, N> invCovSum = covSum.inverse();
        T preFac = 1.0 / GaussianNormalizationFactor(covSum);

        Matrix<T, N, N> invCovP = covP.inverse();
        Matrix<T, N, N> invCovQ = covQ.inverse();

        T newWeight = weightP * weightQ * preFac * std::exp(-0.5 * SquaredMahalonobisDistance(meanP, meanQ, invCovSum));
        Matrix<T, N, N> newCovariance = (invCovP + invCovQ).inverse();
        Vector<T, N> newMean = newCovariance * (invCovP * meanP + invCovQ * meanQ);

        return {newMean, newCovariance, newWeight};
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline GaussianStruct<T, N>
    MultiplyGaussians(const GaussianStruct<T, N> &P, const GaussianStruct<T, N> &Q) {
        return MultiplyGaussians(P.mean, P.covariance, P.weight, Q.mean, Q.covariance, Q.weight);
    }

}

#endif //ENGINE24_GAUSSIANPRODUCT_H
