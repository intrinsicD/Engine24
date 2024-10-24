//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANPDF_H
#define ENGINE24_GAUSSIANPDF_H

#include "GaussianStruct.h"
#include "GaussianMahalonobisDistance.h"
#include "GaussianNormalizationFactor.h"

namespace Bcg{
    template<typename T, int N>
    CUDA_HOST_DEVICE inline T GaussianPdf(const Vector<T, N> &x, const Vector<T, N> &mean, const Matrix<T, N, N> &cov) {
        return std::exp(-0.5 * SquaredMahalonobisDistance(x, mean, cov.inverse())) / GaussianNormalizationFactor(cov);
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline T GaussianPdf(const GaussianStruct<T, N> &P, const Vector<T, N> &point) {
        return GaussianPdf(point, P.mean, P.covariance);
    }
}

#endif //ENGINE24_GAUSSIANPDF_H
