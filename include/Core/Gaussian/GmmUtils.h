#pragma once

#include "Eigen/Dense"
#include "GraphInterface.h"
#include "LaplacianOperator.h"
#include "Logger.h"

#include <chrono>

namespace Bcg {
    class Stopwatch {
    public:
        explicit Stopwatch(const std::string &name) : m_name(name), m_start(std::chrono::high_resolution_clock::now()) {
        }

        ~Stopwatch() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();
            Log::Info(m_name + " took " + std::to_string(duration) + " ms.");
        }

    private:
        std::string m_name;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    };

    /**
     * @brief Evaluates the probability density function (PDF) of a single 3D anisotropic Gaussian.
     * N(x) = (1 / ((2*pi)^(3/2) * sqrt(det(Sigma)))) * exp(-0.5 * (x-mu)^T * Sigma^-1 * (x-mu))
     * @param x The 3D point at which to evaluate the Gaussian.
     * @param mu The 3D mean vector of the Gaussian.
     * @param Sigma The 3x3 covariance matrix of the Gaussian.
     * @param Sigma_inv Precomputed inverse of the covariance matrix for efficiency.
     * @param det_Sigma Precomputed determinant of the covariance matrix for efficiency.
     * @return The scalar value of the Gaussian PDF at point x.
     */
    inline double evaluate_gaussian(const Eigen::Vector3d &x, const Eigen::Vector3d &mu,
                                    const Eigen::Matrix3d &Sigma_inv, double det_Sigma) {
        // Normalization constant
        const double PI = 3.14159265358979323846;
        double norm_const = 1.0 / (std::pow(2 * PI, 1.5) * std::sqrt(det_Sigma));

        // Exponent term
        Eigen::Vector3d x_minus_mu = x - mu;
        double exponent = -0.5 * x_minus_mu.transpose() * Sigma_inv * x_minus_mu;

        return norm_const * std::exp(exponent);
    }

    /**
     * @brief Computes the value of the GMM scalar field F(x) at a given point.
     * F(x) = sum_{i=1 to N} [ w_i * N_i(x) ]
     * @param x The 3D point at which to evaluate the GMM field.
     * @param mus A vector of mean vectors for each Gaussian in the GMM.
     * @param Sigmas A vector of covariance matrices for each Gaussian in the GMM.
     * @param weights A vector of weights for each Gaussian in the GMM.
     * @return The scalar value of the GMM field at point x.
     */
    inline double compute_gmm_field_value(const Eigen::Vector3d &x,
                                          const std::vector<Eigen::Vector3d> &mus,
                                          const std::vector<Eigen::Matrix3d> &Sigmas,
                                          const std::vector<double> &weights) {
        double total_value = 0.0;
        for (size_t i = 0; i < mus.size(); ++i) {
            Eigen::Matrix3d Sigma_inv = Sigmas[i].inverse();
            double det_Sigma = Sigmas[i].determinant();
            if (det_Sigma <= 0) continue;
            double weight = weights.empty() ? 1.0 : weights[i];
            total_value += weight * evaluate_gaussian(x, mus[i], Sigma_inv, det_Sigma);
        }
        return total_value;
    }

    /**
     * @brief Computes the gradient of the GMM scalar field F(x) at a given point.
     * grad F(x) = - sum_{i=1 to N} [ w_i * N_i(x) * Sigma_i^-1 * (x - mu_i) ]
     * @param x The 3D point at which to compute the gradient.
     * @param mus A vector of mean vectors for each Gaussian in the GMM.
     * @param Sigmas A vector of covariance matrices for each Gaussian in the GMM.
     * @param weights A vector of weights for each Gaussian in the GMM.
     * @return The 3D gradient vector of the GMM field at point x.
     */
    inline Eigen::Vector3d compute_gmm_gradient(const Eigen::Vector3d &x,
                                                const std::vector<Eigen::Vector3d> &mus,
                                                const std::vector<Eigen::Matrix3d> &Sigmas,
                                                const std::vector<double> &weights) {
        Eigen::Vector3d total_gradient = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < mus.size(); ++i) {
            Eigen::Matrix3d Sigma_inv = Sigmas[i].inverse();
            double det_Sigma = Sigmas[i].determinant();
            if (det_Sigma <= 0) continue;
            double N_i_at_x = evaluate_gaussian(x, mus[i], Sigma_inv, det_Sigma);
            double weight = weights.empty() ? 1.0 : weights[i];
            Eigen::Vector3d grad_i = weight * N_i_at_x * Sigma_inv * (x - mus[i]);
            total_gradient -= grad_i;
        }
        return total_gradient;
    }

    /**
     * @brief Projects a point onto the GMM level-set surface using Newton's method.
     *
     * @param start_point The initial 3D point to project (e.g., mu_i or mu_ij).
     * @param mus Vector of mean vectors for the GMM.
     * @param Sigmas Vector of covariance matrices for the GMM.
     * @param weights Vector of weights for the GMM.
     * @param iso_value The level-set value 'c' defining the surface.
     * @param iterations The number of Newton steps to perform.
     * @return The projected 3D point on or very near the surface.
     */
    inline Eigen::Vector3d project_point_to_surface(const Eigen::Vector3d &start_point,
                                                    const std::vector<Eigen::Vector3d> &mus,
                                                    const std::vector<Eigen::Matrix3d> &Sigmas,
                                                    const std::vector<double> &weights,
                                                    double iso_value,
                                                    int iterations = 5) {
        Eigen::Vector3d current_point = start_point;

        for (int i = 0; i < iterations; ++i) {
            // 1. Evaluate the field value F(x_k)
            double F_x = compute_gmm_field_value(current_point, mus, Sigmas, weights);

            // 2. Evaluate the gradient grad F(x_k)
            Eigen::Vector3d grad_F_x = compute_gmm_gradient(current_point, mus, Sigmas, weights);

            double grad_norm_sq = grad_F_x.squaredNorm();
            if (grad_norm_sq < 1e-12) {
                // Gradient is too small; cannot proceed reliably.
                // This can happen far from the surface or at a saddle point.
                // Returning the current point is a safe fallback.
                return current_point;
            }

            // 3. Apply the Newton update step
            // x_{k+1} = x_k - (F(x_k) - c) / ||grad F(x_k)||^2 * grad F(x_k)
            double error = F_x - iso_value;
            current_point -= (error / grad_norm_sq) * grad_F_x;
        }

        return current_point;
    }

    /**
     * @brief Computes the unit normal vector of the GMM level-set surface at a point x.
     * The normal is the normalized gradient of the GMM scalar field F(x).
     * n(x) = normalize( grad F(x) )
     * grad F(x) = - sum_{i=1 to N} [ w_i * N_i(x) * Sigma_i^-1 * (x - mu_i) ]
     * @param x The 3D point on the surface.
     * @param mus A vector of 3D mean vectors for each Gaussian.
     * @param Sigmas A vector of 3x3 covariance matrices for each Gaussian.
     * @param weights A vector of scalar weights for each Gaussian.
     * @return The 3D unit normal vector at point x.
     */
    inline Eigen::Vector3d compute_surface_normal(const Eigen::Vector3d &x,
                                                  const std::vector<Eigen::Vector3d> &mus,
                                                  const std::vector<Eigen::Matrix3d> &Sigmas,
                                                  const std::vector<double> &weights) {
        Eigen::Vector3d grad = compute_gmm_gradient(x, mus, Sigmas, weights);
        if (grad.norm() > 1e-9) {
            return grad.normalized();
        }
        return Eigen::Vector3d::UnitZ(); // Fallback
    }

    /**
     * @brief Computes the projection matrix P = I - n*n^T onto the plane defined by a normal.
     * @param normal The 3D normal vector of the plane. Must be a unit vector.
     * @return The 3x3 projection matrix P.
     */
    inline Eigen::Matrix3d compute_projection_matrix_P(const Eigen::Vector3d &normal) {
        // Assumes `normal` is already normalized for efficiency.
        return Eigen::Matrix3d::Identity() - normal * normal.transpose();
    }

    /**
     * @brief Computes an orthonormal basis V = [v1, v2] for the tangent plane.
     * @param normal The 3D normal vector of the plane. Must be a unit vector.
     * @return A 3x2 matrix where columns are the orthonormal basis vectors of the tangent plane.
     */
    inline Eigen::Matrix<double, 3, 2> compute_tangent_basis(const Eigen::Vector3d &normal) {
        // Find a vector that is not parallel to the normal.
        Eigen::Vector3d a = (std::abs(normal.x()) > 0.9) ? Eigen::Vector3d(0, 1, 0) : Eigen::Vector3d(1, 0, 0);

        // Use cross products to find two orthogonal vectors in the plane.
        Eigen::Vector3d v1 = normal.cross(a).normalized();
        Eigen::Vector3d v2 = normal.cross(v1).normalized();

        Eigen::Matrix<double, 3, 2> V;
        V.col(0) = v1;
        V.col(1) = v2;
        return V;
    }

    /**
     * @brief Computes the mean of the product of two Gaussians.
     * mu_ij = (Sigma_i^-1 + Sigma_j^-1)^-1 * (Sigma_i^-1*mu_i + Sigma_j^-1*mu_j)
     * @param mu_i Mean of the first Gaussian.
     * @param Sigma_i Covariance of the first Gaussian.
     * @param mu_j Mean of the second Gaussian.
     * @param Sigma_j Covariance of the second Gaussian.
     * @return The mean vector mu_ij.
     */
    inline Eigen::Vector3d compute_mu_ij(const Eigen::Vector3d &mu_i, const Eigen::Matrix3d &Sigma_i,
                                         const Eigen::Vector3d &mu_j, const Eigen::Matrix3d &Sigma_j) {
        Eigen::Matrix3d Sigma_i_inv = Sigma_i.inverse();
        Eigen::Matrix3d Sigma_j_inv = Sigma_j.inverse();

        Eigen::Matrix3d Sigma_ij = (Sigma_i_inv + Sigma_j_inv).inverse();

        return Sigma_ij * (Sigma_i_inv * mu_i + Sigma_j_inv * mu_j);
    }

    /**
     * @brief Projects a 3D covariance matrix onto a 2D tangent plane.
     * tilde_Sigma = V^T * Sigma * V
     * @param Sigma The 3x3 covariance matrix to project.
     * @param tangent_basis_V The 3x2 matrix whose columns form the basis of the tangent plane.
     * @return The resulting 2x2 projected covariance matrix.
     */
    inline Eigen::Matrix2d compute_tilde_Sigma(const Eigen::Matrix3d &Sigma,
                                               const Eigen::Matrix<double, 3, 2> &tangent_basis_V) {
        return tangent_basis_V.transpose() * Sigma * tangent_basis_V;
    }

    /**
     * @brief Computes the overlap factor K_ij between two Gaussians.
     * K_ij = exp( -0.5 * (mu_i-mu_j)^T * (Sigma_i+Sigma_j)^-1 * (mu_i-mu_j) )
     * @param mu_i Mean of the first Gaussian.
     * @param Sigma_i Covariance of the first Gaussian.
     * @param mu_j Mean of the second Gaussian.
     * @param Sigma_j Covariance of the second Gaussian.
     * @return The scalar overlap factor K_ij.
     */
    inline double compute_K_ij(const Eigen::Vector3d &mu_i, const Eigen::Matrix3d &Sigma_i,
                               const Eigen::Vector3d &mu_j, const Eigen::Matrix3d &Sigma_j) {
        Eigen::Vector3d mu_diff = mu_i - mu_j;
        Eigen::Matrix3d Sigma_sum_inv = (Sigma_i + Sigma_j).inverse();

        double exponent = -0.5 * mu_diff.transpose() * Sigma_sum_inv * mu_diff;

        return std::exp(exponent);
    }

    /**
     * @brief Computes the approximate mass matrix entry M_ij.
     * M_ij approx K_ij * 2*pi * sqrt(det(tilde_Sigma_ij))
     * @param K_ij The precomputed overlap factor.
     * @param tilde_Sigma_ij The 2x2 projected covariance of the product Gaussian.
     * @return The approximate scalar value of M_ij.
     */
    inline double compute_M_ij_approx(double K_ij, const Eigen::Matrix2d &tilde_Sigma_ij) {
        double det = tilde_Sigma_ij.determinant();
        if (det <= 0) {
            // Handle potential numerical issues with non-positive definite matrices
            return 0.0;
        }
        return K_ij * 2.0 * M_PI * std::sqrt(det);
    }

    /**
     * @brief Computes the trace term from the stiffness matrix approximation A_ij.
     * Term = tr( tilde_Sigma_j^-1 * tilde_Sigma_i^-1 * tilde_Sigma_ij )
     * @param tilde_Sigma_i The 2x2 projected covariance of Gaussian i.
     * @param tilde_Sigma_j The 2x2 projected covariance of Gaussian j.
     * @param tilde_Sigma_ij The 2x2 projected covariance of the product Gaussian ij.
     * @return The scalar value of the trace term.
     */
    inline double compute_trace_term_A_ij(const Eigen::Matrix2d &tilde_Sigma_i, const Eigen::Matrix2d &tilde_Sigma_j,
                                          const Eigen::Matrix2d &tilde_Sigma_ij) {
        Eigen::Matrix2d product = tilde_Sigma_j.inverse() * tilde_Sigma_i.inverse() * tilde_Sigma_ij;
        return product.trace();
    }

    /**
     * @brief Computes the projection-dependent term from the stiffness matrix approximation A_ij.
     * Term = (mu_ij-mu_j)^T * Sigma_j^-1 * P * Sigma_i^-1 * (mu_ij-mu_i)
     * @param mu_i Mean of Gaussian i.
     * @param Sigma_i Covariance of Gaussian i.
     * @param mu_j Mean of Gaussian j.
     * @param Sigma_j Covariance of Gaussian j.
     * @param mu_ij Mean of the product Gaussian ij.
     * @param P The 3x3 projection matrix onto the tangent plane at hat_mu_ij.
     * @return The scalar value of the projection term.
     */
    inline double compute_projection_term_A_ij(
        const Eigen::Vector3d &mu_i, const Eigen::Matrix3d &Sigma_i,
        const Eigen::Vector3d &mu_j, const Eigen::Matrix3d &Sigma_j,
        const Eigen::Vector3d &mu_ij, const Eigen::Matrix3d &P) {
        Eigen::Vector3d vec_i = Sigma_i.inverse() * (mu_ij - mu_i);
        Eigen::Vector3d vec_j = Sigma_j.inverse() * (mu_ij - mu_j);

        // The result is a 1x1 matrix, so we extract the scalar value.
        return vec_j.transpose() * P * vec_i;
    }

    /**
     * @brief Precomputes the mean vectors mu_ij for all pairs of Gaussians in the graph.
     * @param gi A reference to the graph interface containing Gaussian data.
     */
    inline LaplacianMatrices BuildGmmApproxLaplacian(GraphInterface &gi, double iso_value = 0.01) {
        Stopwatch total_stopwatch("Total BuildGmmApproxLaplacian");

        LaplacianMatrices matrices;
        auto mu_ij = gi.edge_property<Eigen::Vector3d>("e:mu_ij", Eigen::Vector3d::Zero());
        auto proj_m_ij = gi.edge_property<Eigen::Vector3d>("e:proj_mu_ij", Eigen::Vector3d::Zero());
        auto sigma_ij = gi.edge_property<Eigen::Matrix3d>("e:sigma_ij", Eigen::Matrix3d::Zero());
        auto normal_ij = gi.edge_property<Eigen::Vector3d>("e:normal_ij", Eigen::Vector3d::Zero());
        auto a_ij = gi.edge_property<double>("e:a_ij", 0);
        auto m_ij = gi.edge_property<double>("e:m_ij", 0);
        auto k_ij = gi.edge_property<double>("e:k_ij", 0);
        auto trace_ij = gi.edge_property<double>("e:trace_ij", 0);

        auto m_i = gi.vertex_property<Eigen::Vector3d>("v:mu_i", Eigen::Vector3d::Zero());
        auto scale = gi.vertex_property<Vector<float, 3> >("v:scale", Vector<float, 3>(0.0f));
        auto rotation = gi.vertex_property<Vector<float, 4> >("v:rotation", Vector<float, 4>(0.0f));
        auto sigma_i = gi.vertex_property<Eigen::Matrix3d>("v:sigma_i", Eigen::Matrix3d::Identity());
        auto sigma_i_inverse = gi.vertex_property<Eigen::Matrix3d>("v:sigma_i_inverse", Eigen::Matrix3d::Identity());
        auto a_ii = gi.vertex_property<double>("v:stiffness_diag");
        auto m_ii = gi.vertex_property<double>("v:mass_diag"); {
            Stopwatch sw("Stage 1: Precomputing per-vertex covariances");
            for (const auto v_i: gi.vertices) {
                m_i[v_i] = MapConst(glm::dvec3(gi.vpoint[v_i]));
                auto v_i_quat = Eigen::Quaternion<double>(rotation[v_i].w, rotation[v_i].x, rotation[v_i].y,
                                                          rotation[v_i].z);
                auto v_i_scale = Eigen::Vector3d(scale[v_i].x, scale[v_i].y, scale[v_i].z);
                CovarianceInterface<double> i_cov_i(v_i_scale, v_i_quat);
                sigma_i[v_i] = i_cov_i.get_covariance_matrix();
                sigma_i_inverse[v_i] = sigma_i[v_i].inverse();
            }
        } {
            Stopwatch sw("Stage 2: Precomputing per-edge product Gaussians");
            for (const auto edge: gi.edges) {
                auto h0 = gi.get_halfedge(edge, 0);
                auto v_i = gi.from_vertex(h0);
                auto v_j = gi.to_vertex(h0);

                const Eigen::Vector3d mu_i = m_i[v_i];
                const Eigen::Vector3d mu_j = m_i[v_j];

                const Eigen::Matrix3d &Sigma_i = sigma_i[v_i];
                const Eigen::Matrix3d &Sigma_j = sigma_i[v_j];

                const Eigen::Matrix3d &Sigma_i_inv = sigma_i_inverse[v_i];
                const Eigen::Matrix3d &Sigma_j_inv = sigma_i_inverse[v_j];

                mu_ij[edge] = compute_mu_ij(mu_i, Sigma_i, mu_j, Sigma_j);;
                sigma_ij[edge] = (Sigma_i_inv + Sigma_j_inv).inverse();
            }
        }

        std::vector<Eigen::Triplet<float> > triplets_A;
        std::vector<Eigen::Triplet<float> > triplets_M; {
            Stopwatch sw("Stage 3: Per-edge projection and term evaluation");
            for (const auto edge: gi.edges) {
                auto h0 = gi.get_halfedge(edge, 0);
                auto v_i = gi.from_vertex(h0);
                auto v_j = gi.to_vertex(h0);

                const Eigen::Vector3d mu_i = MapConst(glm::dvec3(gi.vpoint[v_i]));
                const Eigen::Vector3d mu_j = MapConst(glm::dvec3(gi.vpoint[v_j]));

                const Eigen::Vector3d &Mu_ij = mu_ij[edge];
                const Eigen::Matrix3d &Sigma_ij = sigma_ij[edge];

                const Eigen::Matrix3d &Sigma_i = sigma_i[v_i];
                const Eigen::Matrix3d &Sigma_j = sigma_i[v_j];

                proj_m_ij[edge] = project_point_to_surface(mu_ij[edge], m_i.vector(), sigma_i.vector(), {}, iso_value);
                normal_ij[edge] = compute_surface_normal(proj_m_ij[edge], m_i.vector(), sigma_i.vector(), {});

                // 2. Compute the geometric setup at hat_mu_ij
                Eigen::Matrix3d P = compute_projection_matrix_P(normal_ij[edge]);
                Eigen::Matrix<double, 3, 2> V = compute_tangent_basis(normal_ij[edge]);

                // 3. Compute the 2D projected covariances
                Eigen::Matrix2d tilde_Sigma_i = compute_tilde_Sigma(Sigma_i, V);
                Eigen::Matrix2d tilde_Sigma_j = compute_tilde_Sigma(Sigma_j, V);
                Eigen::Matrix2d tilde_Sigma_ij = compute_tilde_Sigma(Sigma_ij, V);

                // 4. Calculate the Galerkin matrix components
                double K_ij = compute_K_ij(mu_i, Sigma_i, mu_j, Sigma_j);
                double M_ij = compute_M_ij_approx(K_ij, tilde_Sigma_ij);

                double trace_term = compute_trace_term_A_ij(tilde_Sigma_i, tilde_Sigma_j, tilde_Sigma_ij);
                double projection_term = compute_projection_term_A_ij(mu_i, Sigma_i, mu_j, Sigma_j, Mu_ij, P);

                // The full approximation for A_ij would be M_ij * (trace_term + projection_term)
                double A_ij = M_ij * (trace_term + projection_term);

                m_ij[edge] = A_ij;
                a_ij[edge] = M_ij;
                k_ij[edge] = K_ij;
                trace_ij[edge] = trace_term;

                double stiffness_val = a_ij[edge];
                double mass_val = m_ij[edge];

                auto v_i_idx = v_i.idx();
                auto v_j_idx = v_j.idx();

                // Add the symmetric off-diagonal entries
                triplets_A.emplace_back(v_i_idx, v_j_idx, stiffness_val);
                triplets_A.emplace_back(v_j_idx, v_i_idx, stiffness_val);

                triplets_M.emplace_back(v_j_idx, v_j_idx, mass_val);
                triplets_M.emplace_back(v_j_idx, v_j_idx, mass_val);

                // Accumulate the sums for the diagonal entries
                a_ii[v_i] -= stiffness_val;
                a_ii[v_j] -= stiffness_val;

                // For the lumped mass matrix, the diagonal is the sum of off-diagonals
                m_ii[v_i] += mass_val;
                m_ii[v_j] += mass_val;
            }
        }

        {
            Stopwatch sw("Stage 4: Assembling sparse matrices from triplets");
            for (const auto v_i: gi.vertices) {
                const auto i = v_i.idx();
                triplets_A.emplace_back(i, i, a_ii[v_i]);

                // For the mass matrix, either use the lumped version (diag_M)
                // or compute a more accurate diagonal term if you have it.
                // Let's stick with the lumped mass for now.
                // M_ii = integral of phi_i^2. A simple approx is 1.0 or sum of row.
                // Let's add the diagonal sum plus a base value (e.g., 1.0) to keep it well-conditioned.
                // A proper M_ii would require projecting and integrating phi_i^2.
                triplets_M.emplace_back(i, i, m_ii[v_i]);
            }
            matrices.build(triplets_A, triplets_M, gi.vertices.n_vertices());
        }

        return matrices;
    }
}
