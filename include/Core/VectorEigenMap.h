//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_VECTOREIGENMAP_H
#define ENGINE24_VECTOREIGENMAP_H

#include "Vector.h"
#include "Eigen/Core"

namespace Bcg::Math {
    template<typename T, int N>
    CUDA_HOST Eigen::Map<Eigen::Matrix<T, N, 1>> EigenMap(Vector<T, N> &v) {
        return Eigen::Map<Eigen::Matrix<T, N, 1>>(v.data(), N);
    }

    template<typename T, int N>
    CUDA_HOST Eigen::Map<const Eigen::Matrix<T, N, 1>> EigenMap(const Vector<T, N> &v) {
        return Eigen::Map<const Eigen::Matrix<T, N, 1>>(v.data(), N);
    }

    template<typename T, int N>
    CUDA_HOST Eigen::Map<Eigen::Matrix<T, N, 1>> EigenMap(Vector<T, N> &&v) {
        return Eigen::Map<Eigen::Matrix<T, N, 1>>(v.data(), N);
    }

    template<typename T, int N>
    CUDA_HOST Eigen::Map<Eigen::Matrix<T, -1, N>> EigenMap(std::vector<Vector<T, N>> &v) {
        return Eigen::Map<Eigen::Matrix<T, -1, N>>(v.data(), v.size(), N);
    }

    template<typename T, int N>
    CUDA_HOST Eigen::Map<const Eigen::Matrix<T, -1, N>> EigenMap(const std::vector<Vector<T, N>> &v) {
        return Eigen::Map<const Eigen::Matrix<T, -1, N>>(v.data(), v.size(), N);
    }

    template<typename T, int N>
    CUDA_HOST Eigen::Map<Eigen::Matrix<T, -1, N>> EigenMap(std::vector<Vector<T, N>> &&v) {
        return Eigen::Map<Eigen::Matrix<T, -1, N>>(v.data(), v.size(), N);
    }
}

#endif //ENGINE24_VECTOREIGENMAP_H
