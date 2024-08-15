//
// Created by alex on 15.08.24.
//

#ifndef ENGINE24_GAUSSIAN_H
#define ENGINE24_GAUSSIAN_H

#include "MatVec.h"
#include "RigidTransform.h"

namespace Bcg {
    template<typename Scalar, int N>
    struct Gaussian {
        Matrix<Scalar, N, N> covariance;
        Vector<Scalar, N> mean;
    };

    template<typename Scalar>
    struct Gaussian<Scalar, 3> {
        Vector<Scalar, 3> mean;
        Matrix<Scalar, 3, 3> covariance;
        Scalar weight;

        Gaussian() : mean(Vector<Scalar, 3>::Zero()), covariance(Matrix<Scalar, 3, 3>::Identity()), weight(1) {

        }

        Gaussian(const Vector<Scalar, 3> &mean, const Matrix<Scalar, 3, 3> &covariance, Scalar weight = 1) : mean(mean),
                                                                                                             covariance(
                                                                                                                     covariance),
                                                                                                             weight(weight) {

        }

        static Gaussian Zero() {
            return Gaussian(Vector<Scalar, 3>::Zero(), Matrix<Scalar, 3, 3>::Zero(), 0);
        }

        Scalar pdf(const Vector<Scalar, 3> &x) const {
            return std::exp(-0.5 * (x - mean).transpose() * covariance.inverse() * (x - mean)) /
                   std::sqrt(std::pow(2 * M_PI, 3) * covariance.determinant());
        }

        Gaussian operator*(const Gaussian &other) const {
            Matrix<Scalar, 3, 3> covSum = covariance + other.covariance;
            Matrix<Scalar, 3, 3> invCovSum = covSum.inverse();
            Scalar preFac = 1.0 / std::sqrt(std::pow(2 * M_PI, 3) * covSum.determinant());

            Matrix<Scalar, 3, 3> invCov0 = covariance.inverse();
            Matrix<Scalar, 3, 3> invCov1 = other.covariance.inverse();
            Vector<Scalar, 3> d = mean - other.mean;

            Gaussian product;
            product.weight = weight * other.weight * preFac * std::exp(-0.5 * d.transpose() * invCovSum * d);
            product.covariance = (invCov0 + invCov1).inverse();
            product.mean = product.covariance * (invCov0 * mean + invCov1 * other.mean);

            return product;
        }
    };

    template<typename Scalar, int N>
    Matrix<Scalar, N, N> RotateCovariance(const Matrix<Scalar, N, N> &cov, const Matrix<Scalar, N, N> &R) {
        return R * cov * R.transpose();
    }

    template<typename Scalar, int N>
    Scalar SquaredMahalonobisDistance(const Vector<Scalar, N> &x,
                                      const Vector<Scalar, N> &y,
                                      const Matrix<Scalar, N, N> &cov) {
        Vector<Scalar, N> d = x - y;
        return d.transpose() * cov.inverse() * d;
    }

    template<typename Scalar, int N>
    Scalar MahalonobisDistance(const Vector<Scalar, N> &x,
                               const Vector<Scalar, N> &y,
                               const Matrix<Scalar, N, N> &cov) {
        return std::sqrt(SquaredMahalonobisDistance(x, y, cov));
    }

    template<typename Scalar>
    Scalar KullbackLeiblerDivergence(const Gaussian<Scalar, 3> &p, const Gaussian<Scalar, 3> &q) {
        Matrix<Scalar, 3, 3> invCovQ = q.covariance.inverse();
        Vector<Scalar, 3> d = p.mean - q.mean;
        Scalar tr = (invCovQ * p.covariance).trace();
        Scalar logDet = std::log(q.covariance.determinant() / p.covariance.determinant());
        return 0.5 * (tr + d.transpose() * invCovQ * d - 3 + logDet);
    }

    template<typename Scalar>
    Scalar BhattacharyyaDistance(const Gaussian<Scalar, 3> &p, const Gaussian<Scalar, 3> &q) {
        Matrix<Scalar, 3, 3> covSum = 0.5 * (p.covariance + q.covariance);
        Vector<Scalar, 3> d = p.mean - q.mean;
        Scalar detCovSum = covSum.determinant();
        Scalar detCovP = p.covariance.determinant();
        Scalar detCovQ = q.covariance.determinant();
        return 0.125 * d.transpose() * covSum.inverse() * d +
               0.5 * std::log(detCovSum / std::sqrt(detCovP * detCovQ));
    }

    template<typename Scalar>
    Gaussian<Scalar, 3> operator*(const RigidTransform &transform, const Gaussian<Scalar, 3> &gaussian) {
        return Gaussian<Scalar, 3>(transform * gaussian.mean,
                                   RotateCovariance(gaussian.covariance, transform.matrix().block<3, 3>(0, 0)),
                                   gaussian.weight);
    }
}

#endif //ENGINE24_GAUSSIAN_H
