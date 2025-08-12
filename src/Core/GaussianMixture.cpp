//
// Created by alex on 12.08.25.
//

#include "GaussianMixture.h"
#include "PropertyEigenMap.h" // For matrix operations, inverse, determinant
#include "GlmToEigen.h" // For matrix operations, inverse, determinant
#include <vector>
#include <random> // For random initialization in fit()
namespace Bcg {
    // Define a constant for 2*PI, which is used frequently in Gaussian PDF calculations.
    const double TWO_PI_D = 2.0 * 3.14159265358979323846;

    /**
     * @brief Implements the Expectation-Maximization (EM) algorithm to fit the GMM to the data.
     * All internal computations are done in double precision for numerical stability.
     */
    void GaussianMixture::fit(const std::vector<PointType> &data, size_t num_gaussians) {
        if (data.empty() || num_gaussians == 0) return;

        const size_t n_points = data.size();
        const size_t n_gaussians = num_gaussians;

        // --- 0. Convert all input data to Eigen double vectors for stable computation ---
        std::vector<Eigen::Vector3d> eigen_data(n_points);
        for(size_t i = 0; i < n_points; ++i) {
            eigen_data[i] = ToEigen(data[i]).cast<double>();
        }

        // --- 1. Initialization (using Eigen double types) ---
        std::vector<Eigen::Vector3d> current_means(n_gaussians);
        std::vector<Eigen::Matrix3d> current_covs(n_gaussians);
        std::vector<double> current_weights(n_gaussians);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, n_points - 1);
        for (size_t k = 0; k < n_gaussians; ++k) {
            current_means[k] = eigen_data[distrib(gen)];
            current_covs[k] = Eigen::Matrix3d::Identity();
            current_weights[k] = 1.0 / n_gaussians;
        }

        // --- 2. EM Iteration ---
        const int max_iters = 100;
        Eigen::MatrixXd responsibilities(n_points, n_gaussians);

        for (int iter = 0; iter < max_iters; ++iter) {
            // --- E-Step ---
            for (long i = 0; i < n_points; ++i) {
                double likelihood_sum = 0.0;
                for (long k = 0; k < n_gaussians; ++k) {
                    const Eigen::Vector3d diff = eigen_data[i] - current_means[k];
                    double det = current_covs[k].determinant();
                    if (det < 1e-9) det = 1e-9;
                    Eigen::Matrix3d inv_cov = current_covs[k].inverse();
                    double exponent = -0.5 * diff.transpose() * inv_cov * diff;
                    double pdf_val = (1.0 / std::sqrt(std::pow(TWO_PI_D, 3) * det)) * std::exp(exponent);
                    double weighted_pdf = current_weights[k] * pdf_val;
                    responsibilities(i, k) = weighted_pdf;
                    likelihood_sum += weighted_pdf;
                }
                if (likelihood_sum > 1e-9) {
                    responsibilities.row(i) /= likelihood_sum;
                }
            }
            // --- M-Step ---
            for (size_t k = 0; k < n_gaussians; ++k) {
                double Nk = responsibilities.col(k).sum();
                if (Nk < 1e-9) continue;

                current_weights[k] = Nk / n_points;

                Eigen::Vector3d new_mean = Eigen::Vector3d::Zero();
                for (size_t i = 0; i < n_points; ++i) {
                    new_mean += responsibilities(i, k) * eigen_data[i];
                }
                current_means[k] = new_mean / Nk;

                Eigen::Matrix3d new_cov = Eigen::Matrix3d::Zero();
                for (size_t i = 0; i < n_points; ++i) {
                    Eigen::Vector3d diff = eigen_data[i] - current_means[k];
                    new_cov += responsibilities(i, k) * diff * diff.transpose();
                }
                current_covs[k] = (new_cov / Nk) + (Eigen::Matrix3d::Identity() * 1e-6); // Regularization
            }
        }

        // --- 3. Finalization: Convert back from Eigen double to native float types ---
        std::vector<PointType> final_means(n_gaussians);
        std::vector<Matrix<float, 3, 3>> final_covs(n_gaussians);
        std::vector<float> final_weights(n_gaussians);

        for(size_t k = 0; k < n_gaussians; ++k) {
            final_means[k] = FromEigen(current_means[k]);
            final_covs[k] = FromEigen(current_covs[k]);
            final_weights[k] = static_cast<float>(current_weights[k]);
        }

        set_means(final_means);
        set_covs(final_covs);
        set_weights(final_weights);
    }

    float GaussianMixture::pdf(const Vector<float, 3> &x) const {
        // Convert input to Eigen for computation
        Eigen::Vector3f x_eigen = ToEigen(x);
        double total_pdf = 0.0;

        for (long i = 0; i < means.vector().size(); ++i) {
            Vertex k(i);
            // Convert stored parameters to Eigen for computation
            Eigen::Vector3f mean_k = ToEigen(means[k]);
            Eigen::Matrix3f cov_k = ToEigen(covariances[k]);
            float weight_k = weights[k];

            const Eigen::Vector3f diff = x_eigen - mean_k;
            float det = cov_k.determinant();
            if (det <= 1e-9f) continue;

            Eigen::Matrix3f inv_cov = cov_k.inverse();
            float exponent = -0.5f * diff.transpose() * inv_cov * diff;
            float pdf_val = (1.0f / std::sqrt(std::pow(TWO_PI_D, 3.0) * det)) * std::exp(exponent);

            total_pdf += weight_k * pdf_val;
        }
        return static_cast<float>(total_pdf);
    }

    Vector<float, 3> GaussianMixture::gradient(const Vector<float, 3> &x) const {
        Eigen::Vector3f x_eigen = ToEigen(x);
        Eigen::Vector3f total_gradient = Eigen::Vector3f::Zero();

        for (long k_idx = 0; k_idx < means.vector().size(); ++k_idx) {
            Vertex k(k_idx);
            Eigen::Vector3f mean_k = ToEigen(means[k]);
            Eigen::Matrix3f cov_k = ToEigen(covariances[k]);
            float weight_k = weights[k];

            const Eigen::Vector3f diff = x_eigen - mean_k;
            float det = cov_k.determinant();
            if (det <= 1e-9f) continue;

            Eigen::Matrix3f inv_cov = cov_k.inverse();
            float exponent = -0.5f * diff.transpose() * inv_cov * diff;
            float pdf_val = (1.0f / std::sqrt(std::pow(TWO_PI_D, 3.0) * det)) * std::exp(exponent);

            Eigen::Vector3f grad_val = pdf_val * (-inv_cov * diff);
            total_gradient += weight_k * grad_val;
        }
        return FromEigen(total_gradient);
    }

    Matrix<float, 3, 3> GaussianMixture::hessian(const Vector<float, 3> &x) const {
        Eigen::Vector3f x_eigen = ToEigen(x);
        Eigen::Matrix3f total_hessian = Eigen::Matrix3f::Zero();

        for (long k_idx = 0; k_idx < means.vector().size(); ++k_idx) {
            Vertex k(k_idx);
            Eigen::Vector3f mean_k = ToEigen(means[k]);
            Eigen::Matrix3f cov_k = ToEigen(covariances[k]);
            float weight_k = weights[k];

            const Eigen::Vector3f diff = x_eigen - mean_k;
            float det = cov_k.determinant();
            if (det <= 1e-9f) continue;

            Eigen::Matrix3f inv_cov = cov_k.inverse();
            float exponent = -0.5f * diff.transpose() * inv_cov * diff;
            float pdf_val = (1.0f / std::sqrt(std::pow(TWO_PI_D, 3.0) * det)) * std::exp(exponent);

            Eigen::Vector3f term = inv_cov * diff;
            Eigen::Matrix3f hess_val = pdf_val * (term * term.transpose() - inv_cov);
            total_hessian += weight_k * hess_val;
        }
        return FromEigen(total_hessian);
    }

    Vector<float, 3> GaussianMixture::normal(const Vector<float, 3> &x) const {
        Vector<float, 3> grad = gradient(x);
        float norm_val = length(grad);
        if (norm_val > 1e-9f) {
            return grad / norm_val;
        } else {
            return Vector<float, 3>(0.0f, 0.0f, 0.0f);
        }
    }

    Matrix<float, 3, 3> GaussianMixture::ortho_projector(const Vector<float, 3> &x) const {
        Vector<float, 3> n = normal(x);
        // Assuming your Matrix class supports this constructor for identity
        // and your Vector type supports outer product or equivalent multiplication
        return Matrix<float, 3, 3>(1.0f) - outerProduct(n, n);
    }

    Matrix<float, 3, 3> GaussianMixture::second_fundamental_form(const Vector<float, 3> &x) const {
        Vector<float, 3> grad = gradient(x);
        float grad_norm = length(grad);
        if (grad_norm < 1e-9f) {
            return Matrix<float, 3, 3>(); // Return zero/default matrix
        }
        Matrix<float, 3, 3> hess = hessian(x);
        Matrix<float, 3, 3> p = ortho_projector(x);
        return -p * hess * p / grad_norm;
    }
}