//
// Created by alex on 15.08.24.
//

#ifndef ENGINE24_GAUSSIAN_H
#define ENGINE24_GAUSSIAN_H

#include "MatVec.h"
#include "Statistics.h"
#include "RigidTransform.h"
#include "Covariance.h"

namespace Bcg {
    template<typename Scalar, int N>
    struct GaussianStructOfArrays {
        std::vector<Vector<Scalar, N>> means;
        std::vector<Matrix<Scalar, N, N>> covs;
        std::vector<Scalar> weights;
    };

    template<typename Scalar, int N>
    struct IGaussian;

    template<typename Scalar, int N>
    struct Gaussian;

    template<typename Scalar, int N>
    struct GaussianView;

    // Utility functions for common Gaussian operations
    template<typename Scalar, int N>
    Matrix<Scalar, N, N> RotateCovariance(const Matrix<Scalar, N, N> &cov,
                                          const Matrix<Scalar, N, N> &R) {
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

    template<typename Scalar, int N>
    Scalar GaussianPDF(const Vector<Scalar, N> &x,
                       const Vector<Scalar, N> &mean,
                       const Matrix<Scalar, N, N> &covariance) {
        return std::exp(-0.5 * (x - mean).transpose() * covariance.inverse() * (x - mean)) /
               std::sqrt(std::pow(2 * M_PI, N) * covariance.determinant());
    }

    template<typename Scalar, int N>
    Scalar KullbackLeiblerDiv(const Vector<Scalar, N> &meanP, const Matrix<Scalar, N, N> &covP,
                              const Vector<Scalar, N> &meanQ, const Matrix<Scalar, N, N> &covQ) {
        Matrix<Scalar, N, N> invCovQ = covQ.inverse();
        Vector<Scalar, N> d = meanP - meanQ;
        Scalar tr = (invCovQ * covP).trace();
        Scalar logDet = std::log(covQ.determinant() / covP.determinant());
        return 0.5 * (tr + d.transpose() * invCovQ * d - N + logDet);
    }

    template<typename Scalar, int N>
    Scalar BhattacharyyaDist(const Vector<Scalar, N> &meanP, const Matrix<Scalar, N, N> &covP,
                             const Vector<Scalar, N> &meanQ, const Matrix<Scalar, N, N> &covQ) {
        Matrix<Scalar, N, N> covSum = 0.5 * (covP + covQ);
        Vector<Scalar, N> d = meanP - meanQ;
        Scalar detCovSum = covSum.determinant();
        Scalar detCovP = covP.determinant();
        Scalar detCovQ = covQ.determinant();
        return 0.125 * d.transpose() * covSum.inverse() * d +
               0.5 * std::log(detCovSum / std::sqrt(detCovP * detCovQ));
    }

    template<typename Scalar, int N>
    Gaussian<Scalar, N>
    MultiplyGaussians(const Vector<Scalar, N> &meanP, const Matrix<Scalar, N, N> &covP, Scalar weightP,
                      const Vector<Scalar, N> &meanQ, const Matrix<Scalar, N, N> &covQ, Scalar weightQ) {
        Matrix<Scalar, N, N> covSum = covP + covQ;
        Matrix<Scalar, N, N> invCovSum = covSum.inverse();
        Scalar preFac = 1.0 / std::sqrt(std::pow(2 * M_PI, N) * covSum.determinant());

        Matrix<Scalar, N, N> invCovP = covP.inverse();
        Matrix<Scalar, N, N> invCovQ = covQ.inverse();
        Vector<Scalar, N> d = meanP - meanQ;

        Scalar newWeight = weightP * weightQ * preFac * std::exp(-0.5 * d.transpose() * invCovSum * d);
        Matrix<Scalar, N, N> newCovariance = (invCovP + invCovQ).inverse();
        Vector<Scalar, N> newMean = newCovariance * (invCovP * meanP + invCovQ * meanQ);

        return Gaussian<Scalar, N>(newMean, newCovariance, newWeight);
    }

    template<typename Scalar, int N>
    struct IGaussian {
        virtual void build(const std::vector<Vector<Scalar, N>> &points) = 0;

        virtual Scalar pdf(const Vector<Scalar, N> &x) const = 0;

        virtual Scalar kullback_leibler_div(const Gaussian<Scalar, N> &other) const = 0;

        virtual Scalar kullback_leibler_div(const GaussianView<Scalar, N> &other) const = 0;

        virtual Scalar bhattacharyya_dist(const Gaussian<Scalar, N> &other) const = 0;

        virtual Scalar bhattacharyya_dist(const GaussianView<Scalar, N> &other) const = 0;

        virtual Gaussian<Scalar, N> operator*(const Gaussian<Scalar, N> &other) const = 0;

        virtual Gaussian<Scalar, N> operator*(const GaussianView<Scalar, N> &other) const = 0;
    };

    template<typename Scalar, int N>
    struct Gaussian : public IGaussian<Scalar, N> {
        Matrix<Scalar, N, N> covariance;
        Vector<Scalar, N> mean;
        Scalar weight;

        Gaussian() : covariance(Matrix<Scalar, N, N>::Identity()), mean(Vector<Scalar, N>::Zero()), weight(1) {}

        Gaussian(const Vector<Scalar, N> &mean, const Matrix<Scalar, N, N> &covariance, Scalar weight)
                : covariance(covariance), mean(mean), weight(weight) {}

        void build(const std::vector<Vector<Scalar, N>> &points) override {
            mean = Mean(points);
            covariance = Covariance(points, mean);
        }

        Scalar pdf(const Vector<Scalar, N> &x) const override {
            return GaussianPDF(x, mean, covariance);
        }

        Scalar kullback_leibler_div(const Gaussian<Scalar, N> &other) const override {
            return KullbackLeiblerDiv(mean, covariance, other.mean, other.covariance);
        }

        Scalar kullback_leibler_div(const GaussianView<Scalar, N> &other) const override {
            return KullbackLeiblerDiv(mean, covariance, other.mean, other.covariance);
        }

        Scalar bhattacharyya_dist(const Gaussian<Scalar, N> &other) const override {
            return BhattacharyyaDist(mean, covariance, other.mean, other.covariance);
        }

        Scalar bhattacharyya_dist(const GaussianView<Scalar, N> &other) const override {
            return BhattacharyyaDist(mean, covariance, other.mean, other.covariance);
        }

        Gaussian<Scalar, N> operator*(const Gaussian<Scalar, N> &other) const override {
            return MultiplyGaussians(mean, covariance, weight, other.mean, other.covariance, other.weight);
        }

        Gaussian<Scalar, N> operator*(const GaussianView<Scalar, N> &other) const override {
            return MultiplyGaussians(mean, covariance, weight, other.mean, other.covariance, other.weight);
        }
    };

    template<typename Scalar, int N>
    struct GaussianView : public IGaussian<Scalar, N> {
        Matrix<Scalar, N, N> &covariance;
        Vector<Scalar, N> &mean;
        Scalar &weight;

        GaussianView(Vector<Scalar, N> &mean, Matrix<Scalar, N, N> &covariance, Scalar &weight)
                : covariance(covariance), mean(mean), weight(weight) {}

        void build(const std::vector<Vector<Scalar, N>> &points) override {
            mean = Mean(points);
            covariance = Covariance(points, mean);
        }

        Scalar pdf(const Vector<Scalar, N> &x) const override {
            return GaussianPDF(x, mean, covariance);
        }

        Scalar kullback_leibler_div(const Gaussian<Scalar, N> &other) const override {
            return KullbackLeiblerDiv(mean, covariance, other.mean, other.covariance);
        }

        Scalar kullback_leibler_div(const GaussianView<Scalar, N> &other) const override {
            return KullbackLeiblerDiv(mean, covariance, other.mean, other.covariance);
        }

        Scalar bhattacharyya_dist(const Gaussian<Scalar, N> &other) const override {
            return BhattacharyyaDist(mean, covariance, other.mean, other.covariance);
        }

        Scalar bhattacharyya_dist(const GaussianView<Scalar, N> &other) const override {
            return BhattacharyyaDist(mean, covariance, other.mean, other.covariance);
        }

        Gaussian<Scalar, N> operator*(const Gaussian<Scalar, N> &other) const override {
            return MultiplyGaussians(mean, covariance, weight, other.mean, other.covariance, other.weight);
        }

        Gaussian<Scalar, N> operator*(const GaussianView<Scalar, N> &other) const override {
            return MultiplyGaussians(mean, covariance, weight, other.mean, other.covariance, other.weight);
        }
    };

    template<typename Scalar>
    using Gaussian3 = Gaussian<Scalar, 3>;

    template<typename Scalar>
    using GaussianView3 = GaussianView<Scalar, 3>;
}

#endif //ENGINE24_GAUSSIAN_H
