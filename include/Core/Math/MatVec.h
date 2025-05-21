//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATVEC_H
#define ENGINE24_MATVEC_H

#include "Eigen/Core"

namespace Bcg {
    template<typename Derived>
    Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    RandomLike(const Eigen::MatrixBase<Derived> &matrix) {
        return Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Random(
                matrix.derived().rows(), matrix.derived().cols());
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    ZerosLike(const Eigen::MatrixBase<Derived> &matrix) {
        return Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Zero(
                matrix.derived().rows(), matrix.derived().cols());
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    OnesLike(const Eigen::MatrixBase<Derived> &matrix) {
        return Eigen::Matrix<typename Derived::T, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>::Ones(
                matrix.derived().rows(), matrix.derived().cols());
    }
}

#endif //ENGINE24_MATVEC_H
