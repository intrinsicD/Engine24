//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_STATISTICS_H
#define ENGINE24_STATISTICS_H

#include "MatVec.h"

namespace Bcg{
    template<typename Scalar, int N>
    Vector<Scalar, N> Mean(const std::vector<Vector<Scalar, N>> &points) {
        Vector<Scalar, N> mean = Vector<Scalar, N>::Zero();
        for (const Vector<Scalar, N> &point: points) {
            mean += point;
        }
        mean /= points.size();
        return mean;
    }

    template<typename Scalar, int N>
    Matrix<Scalar, N, N> Covariance(const std::vector<Vector<Scalar, N>> &points, const Vector<Scalar, N> &mean) {
        Matrix<Scalar, N, N> cov = Matrix<Scalar, N, N>::Zero();
        for (const Vector<Scalar, N> &point: points) {
            Vector<Scalar, N> d = point - mean;
            cov += d * d.transpose();
        }
        cov /= points.size();
        return cov;
    }

    template<typename Scalar, int N>
    Matrix<Scalar, N, N>
    CovarianceUnbiased(const std::vector<Vector<Scalar, N>> &points, const Vector<Scalar, N> &mean) {
        Matrix<Scalar, N, N> cov = Matrix<Scalar, N, N>::Zero();
        for (const Vector<Scalar, N> &point: points) {
            Vector<Scalar, N> d = point - mean;
            cov += d * d.transpose();
        }
        cov /= (points.size() - 1);
        return cov;
    }
}

#endif //ENGINE24_STATISTICS_H
