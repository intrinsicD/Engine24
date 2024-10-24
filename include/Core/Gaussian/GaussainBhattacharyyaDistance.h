//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSAINBHATTACHARYYADISTANCE_H
#define ENGINE24_GAUSSAINBHATTACHARYYADISTANCE_H

#include "GaussianMahalonobisDistance.h"

namespace Bcg {
    template<typename T, int N>
    CUDA_HOST_DEVICE inline T BhattacharyyaDist(const Vector<T, N> &meanP, const Matrix<T, N, N> &covP,
                               const Vector<T, N> &meanQ, const Matrix<T, N, N> &covQ) {
        Matrix<T, N, N> covSum = 0.5 * (covP + covQ);
        T detCovSum = covSum.determinant();
        T detCovP = covP.determinant();
        T detCovQ = covQ.determinant();
        return 0.125 * SquaredMahalonobisDistance(meanP, meanQ, covSum.inverse()) +
               0.5 * std::log(detCovSum / std::sqrt(detCovP * detCovQ));
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline T BhattacharyyaDist(const GaussianStruct<T, N> &P, const GaussianStruct<T, N> &Q) {
        return BhattacharyyaDist(P.mean, P.covariance, Q.mean, Q.covariance);
    }
}

#endif //ENGINE24_GAUSSAINBHATTACHARYYADISTANCE_H
