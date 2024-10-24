//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANFITTING_H
#define ENGINE24_GAUSSIANFITTING_H

#include "GaussianStruct.h"

namespace Bcg {


    template<typename T, int N>
    CUDA_HOST_DEVICE inline GaussianStruct<T, N>
    GaussianLeastSquaresFitWeighted(const Matrix<T, -1, N> points, const Vector<T, -1> &weights) {
        Vector<T, N> mean = (points.transpose() * weights) / weights.sum();
        Matrix<T, -1, N> points_centered = points.rowwise() - mean.transpose();
        Matrix<T, N, N> cov =
                (points_centered.array().rowwise() * weights.transpose().array()).transpose() * points_centered;
        cov /= weights.sum(); //unbiased estimator
        return {mean, cov, 1};
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline GaussianStruct<T, N> GaussianLeastSquaresFit(const Matrix<T, -1, N> points) {
        return GaussianLeastSquaresFitWeighted(points, Vector<T, -1>::Ones(points.rows()));
    }
}

#endif //ENGINE24_GAUSSIANFITTING_H
