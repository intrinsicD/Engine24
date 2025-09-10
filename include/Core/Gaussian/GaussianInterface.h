#pragma once

#include "CovarianceInterface.h"

namespace Bcg {
    template<typename T>
    struct GaussianInterface {
        Eigen::Vector<T, 3> &mean;
        CovarianceInterface<T> covariance;
        U &weight;

        T pdf(const Eigen::Vector<T, 3> &x) const {

        }
        Eigen::Vector<T, 3> gradient(const Eigen::Vector<T, 3> &x) const {

        }
        Eigen::Matrix<T, 3, 3> hessian(const Eigen::Vector<T, 3> &x) const {

        }

    };
}