//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_GAUSSIANMAHALONOBISDISTANCE_H
#define ENGINE24_GAUSSIANMAHALONOBISDISTANCE_H

#include "GaussianStruct.h"

namespace Bcg {
    template<typename T, int N>
    CUDA_HOST_DEVICE inline T SquaredMahalonobisDistance(const Vector<T, N> &x,
                                        const Vector<T, N> &y,
                                        const Matrix<T, N, N> &invCov) {
        Vector<T, N> d = x - y;
        return d.transpose() * invCov * d;
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE inline T SquaredMahalonobisDistance(const GaussianStruct<T, N> &P,
                                        const Vector<T, N> &point) {
        return SquaredMahalonobisDistance(point, P.mean, P.covariance.inverse());
    }
}

#endif //ENGINE24_GAUSSIANMAHALONOBISDISTANCE_H
