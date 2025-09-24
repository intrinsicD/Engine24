#pragma once

#include "GraphInterface.h"
#include "LaplacianOperator.h"
#include "CovarianceInterface.h"
#include "Logger.h"
#include "AABB.h"
#include "MatTraits.h"

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
    template<typename T>
    inline T evaluate_gaussian(const Vector<T, 3> &x, const Vector<T, 3> &mu,
                               const Matrix<T, 3, 3> &Sigma_inv, T det_Sigma) {
        // Normalization constant
        const T PI = 3.14159265358979323846;
        T norm_const = 1.0 / (std::pow(2 * PI, 1.5) * std::sqrt(det_Sigma));

        // Exponent term
        Vector<T, 3> x_minus_mu = x - mu;
        T exponent = -0.5 * MatTraits<Matrix<T, 3, 3> >::transpose(x_minus_mu) * Sigma_inv * x_minus_mu;

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
    template<typename T>
    inline T compute_gmm_field_value(const Vector<T, 3> &x,
                                     const std::vector<Vector<T, 3> > &mus,
                                     const std::vector<Matrix<T, 3, 3> > &Sigmas,
                                     const std::vector<T> &weights) {
        T total_value = 0.0;
        for (size_t i = 0; i < mus.size(); ++i) {
            Matrix<T, 3, 3> Sigma_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigmas[i]);
            T det_Sigma = MatTraits<Matrix<T, 3, 3> >::determinant(Sigmas[i]);
            if (det_Sigma <= 0) continue;
            T weight = weights.empty() ? 1.0 : weights[i];
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
    template<typename T>
    inline Vector<T, 3> compute_gmm_gradient(const Vector<T, 3> &x,
                                             const std::vector<Vector<T, 3> > &mus,
                                             const std::vector<Matrix<T, 3, 3> > &Sigmas,
                                             const std::vector<T> &weights) {
        Vector<T, 3> total_gradient = Vector<T, 3>::Zero();
        for (size_t i = 0; i < mus.size(); ++i) {
            Matrix<T, 3, 3> Sigma_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigmas[i]);
            T det_Sigma = MatTraits<Matrix<T, 3, 3> >::determinant(Sigmas[i]);
            if (det_Sigma <= 0) continue;
            T N_i_at_x = evaluate_gaussian(x, mus[i], Sigma_inv, det_Sigma);
            T weight = weights.empty() ? 1.0 : weights[i];
            Vector<T, 3> grad_i = weight * N_i_at_x * Sigma_inv * (x - mus[i]);
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
    template<typename T>
    inline Vector<T, 3> project_point_to_surface(const Vector<T, 3> &start_point,
                                                 const std::vector<Vector<T, 3> > &mus,
                                                 const std::vector<Matrix<T, 3, 3> > &Sigmas,
                                                 const std::vector<T> &weights,
                                                 T iso_value,
                                                 int iterations = 5) {
        Vector<T, 3> current_point = start_point;

        for (int i = 0; i < iterations; ++i) {
            // 1. Evaluate the field value F(x_k)
            T F_x = compute_gmm_field_value(current_point, mus, Sigmas, weights);

            // 2. Evaluate the gradient grad F(x_k)
            Vector<T, 3> grad_F_x = compute_gmm_gradient(current_point, mus, Sigmas, weights);

            T grad_norm_sq = VecTraits<Vector<T, 3> >::squared_length(grad_F_x);
            if (grad_norm_sq < 1e-12) {
                // Gradient is too small; cannot proceed reliably.
                // This can happen far from the surface or at a saddle point.
                // Returning the current point is a safe fallback.
                return current_point;
            }

            // 3. Apply the Newton update step
            // x_{k+1} = x_k - (F(x_k) - c) / ||grad F(x_k)||^2 * grad F(x_k)
            T error = F_x - iso_value;
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
    template<typename T>
    inline Vector<T, 3> compute_surface_normal(const Vector<T, 3> &x,
                                               const std::vector<Vector<T, 3> > &mus,
                                               const std::vector<Matrix<T, 3, 3> > &Sigmas,
                                               const std::vector<T> &weights) {
        Vector<T, 3> grad = compute_gmm_gradient(x, mus, Sigmas, weights);
        if (grad.norm() > 1e-9) {
            return grad.normalized();
        }
        return Vector<T, 3>::UnitZ(); // Fallback
    }

    /**
     * @brief Computes the projection matrix P = I - n*n^T onto the plane defined by a normal.
     * @param normal The 3D normal vector of the plane. Must be a unit vector.
     * @return The 3x3 projection matrix P.
     */
    template<typename T>
    inline Matrix<T, 3, 3> compute_projection_matrix_P(const Vector<T, 3> &normal) {
        // Assumes `normal` is already normalized for efficiency.
        return MatTraits<Matrix<T, 3, 3> >::identity() - VecTraits<Vector<T, 3> >::outer_product(normal, normal);
    }

    /**
     * @brief Computes an orthonormal basis V = [v1, v2] for the tangent plane.
     * @param normal The 3D normal vector of the plane. Must be a unit vector.
     * @return A 3x2 matrix where columns are the orthonormal basis vectors of the tangent plane.
     */
    template<typename T>
    inline Eigen::Matrix<T, 3, 2> compute_tangent_basis(const Vector<T, 3> &normal) {
        // Find a vector that is not parallel to the normal.
        Vector<T, 3> a = (std::abs(normal.x()) > 0.9) ? Vector<T, 3>(0, 1, 0) : Vector<T, 3>(1, 0, 0);

        // Use cross products to find two orthogonal vectors in the plane.
        Vector<T, 3> v1 = VecTraits<Vector<T, 3> >::normalize(VecTraits<Vector<T, 3> >::cross(normal, a));
        Vector<T, 3> v2 = VecTraits<Vector<T, 3> >::normalize(VecTraits<Vector<T, 3> >::cross(normal, v1));

        Matrix<T, 3, 2> V;
        V[0] = v1;
        V[1] = v2;
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
    template<typename T>
    inline Vector<T, 3> compute_mu_ij(const Vector<T, 3> &mu_i, const Matrix<T, 3, 3> &Sigma_i,
                                      const Vector<T, 3> &mu_j, const Matrix<T, 3, 3> &Sigma_j) {
        Matrix<T, 3, 3> Sigma_i_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_i);
        Matrix<T, 3, 3> Sigma_j_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_j);

        Matrix<T, 3, 3> Sigma_ij = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_i_inv + Sigma_j_inv);

        return Sigma_ij * (Sigma_i_inv * mu_i + Sigma_j_inv * mu_j);
    }

    /**
     * @brief Projects a 3D covariance matrix onto a 2D tangent plane.
     * tilde_Sigma = V^T * Sigma * V
     * @param Sigma The 3x3 covariance matrix to project.
     * @param tangent_basis_V The 3x2 matrix whose columns form the basis of the tangent plane.
     * @return The resulting 2x2 projected covariance matrix.
     */
    template<typename T>
    inline Matrix<T, 2, 2> compute_tilde_Sigma(const Matrix<T, 3, 3> &Sigma,
                                               const Eigen::Matrix<T, 3, 2> &tangent_basis_V) {
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
    template<typename T>
    inline T compute_K_ij(const Vector<T, 3> &mu_i, const Matrix<T, 3, 3> &Sigma_i,
                          const Vector<T, 3> &mu_j, const Matrix<T, 3, 3> &Sigma_j) {
        Vector<T, 3> mu_diff = mu_i - mu_j;
        Matrix<T, 3, 3> Sigma_sum_inv = (Sigma_i + Sigma_j).inverse();

        T exponent = -0.5 * mu_diff.transpose() * Sigma_sum_inv * mu_diff;

        return std::exp(exponent);
    }

    /**
     * @brief Computes the approximate mass matrix entry M_ij.
     * M_ij approx K_ij * 2*pi * sqrt(det(tilde_Sigma_ij))
     * @param K_ij The precomputed overlap factor.
     * @param tilde_Sigma_ij The 2x2 projected covariance of the product Gaussian.
     * @return The approximate scalar value of M_ij.
     */
    template<typename T>
    inline T compute_M_ij_approx(T K_ij, const Matrix<T, 2, 2> &tilde_Sigma_ij) {
        T det = tilde_Sigma_ij.determinant();
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
    template<typename T>
    inline T compute_trace_term_A_ij(const Matrix<T, 2, 2> &tilde_Sigma_i, const Matrix<T, 2, 2> &tilde_Sigma_j,
                                     const Matrix<T, 2, 2> &tilde_Sigma_ij) {
        Matrix<T, 2, 2> product = tilde_Sigma_j.inverse() * tilde_Sigma_i.inverse() * tilde_Sigma_ij;
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
    template<typename T>
    inline T compute_projection_term_A_ij(
        const Vector<T, 3> &mu_i, const Matrix<T, 3, 3> &Sigma_i,
        const Vector<T, 3> &mu_j, const Matrix<T, 3, 3> &Sigma_j,
        const Vector<T, 3> &mu_ij, const Matrix<T, 3, 3> &P) {
        Vector<T, 3> vec_i = Sigma_i.inverse() * (mu_ij - mu_i);
        Vector<T, 3> vec_j = Sigma_j.inverse() * (mu_ij - mu_j);

        // The result is a 1x1 matrix, so we extract the scalar value.
        return vec_j.transpose() * P * vec_i;
    }

    template<typename T>
    Matrix<T, 3, 3> get_covariance_matrix(const Vector<T, 3> &scale, const Vector<T, 4> &quat) {
        Matrix<T, 3, 3> rotation_matrix = glm::mat3_cast(glm::quat(quat));
        Matrix<T, 3, 3> scale_matrix = Matrix<T, 3, 3>(0.0f);
        scale_matrix[0][0] = scale[0] * scale[0];
        scale_matrix[1][1] = scale[1] * scale[1];
        scale_matrix[2][2] = scale[2] * scale[2];
        return rotation_matrix * scale_matrix * glm::transpose(rotation_matrix);
    }

    template<typename T>
    Matrix<T, 3, 3> get_covariance_matrix(const Vector<T, 3> &scale) {
        Matrix<T, 3, 3> scale_matrix = Matrix<T, 3, 3>(0.0f);
        scale_matrix[0][0] = scale[0] * scale[0];
        scale_matrix[1][1] = scale[1] * scale[1];
        scale_matrix[2][2] = scale[2] * scale[2];
        return scale_matrix;
    }

    template<typename T>
    std::vector<Matrix<T, 3, 3> > compute_covs_from(const std::vector<Vector<T, 3> > &scales) {
        size_t size = scales.size();
        std::vector<Matrix<T, 3, 3> > covariances(size);
        for (size_t i = 0; i < size; i++) {
            covariances[i] = get_covariance_matrix(scales[i]);
        }
        return covariances;
    }

    template<typename T>
    std::vector<Matrix<T, 3, 3> > compute_covs_from(const std::vector<Vector<T, 3> > &scales,
                                                    const std::vector<Vector<T, 4> > &rotations) {
        size_t size = scales.size();
        if (size != rotations.size()) {
            Log::Error("compute_covs_from(): scales.size() != rotations.size()");
            return std::vector<Matrix<T, 3, 3> >(size, Matrix<T, 3, 3>(1.0f));
        }

        std::vector<Matrix<T, 3, 3> > covariances(size);
        for (size_t i = 0; i < size; i++) {
            covariances[i] = get_covariance_matrix(scales[i], rotations[i]);
        }
        return covariances;
    }

    template<typename T>
    std::vector<Matrix<T, 3, 3> > compute_covs_inverse_from(const std::vector<Matrix<T, 3, 3> > &covariances) {
        size_t size = covariances.size();
        std::vector<Matrix<T, 3, 3> > covariances_inv(size);
        for (size_t i = 0; i < size; i++) {
            covariances_inv[i] = glm::inverse(covariances[i]);
        }
        return covariances_inv;
    }

    template<typename T>
    AABB<T> aabb_for(const Vector<T, 3> &mean,
                     const Matrix<T, 3, 3> &covariance) {
        // The diagonal elements of the covariance matrix are the variances along the corresponding axes.
        // The standard deviation is the square root of the variance.
        T sigmaX = std::sqrt(covariance[0][0]);
        T sigmaY = std::sqrt(covariance[1][1]);
        T sigmaZ = std::sqrt(covariance[2][2]);

        // The 3-sigma range gives us the extents of the AABB from the mean.
        Vector<T, 3> extents(3.0f * sigmaX, 3.0f * sigmaY, 3.0f * sigmaZ);

        // The AABB is centered at the Gaussian's mean.
        AABB<T> box;
        box.min = mean - extents;
        box.max = mean + extents;

        return box;
    }

    template<typename T>
    std::vector<AABB<T> > compute_aabbs_from(const std::vector<Vector<T, 3> > &means, const std::vector<Matrix<T, 3, 3> > &covs) {
        size_t size = means.size();
        std::vector<AABB<T> > aabbs(size);
        for (size_t i = 0; i < size; i++) {
            aabbs[i] = aabb_for(means[i], covs[i]);
        }
        return aabbs;
    }


    /**
     * @brief Precomputes the mean vectors mu_ij for all pairs of Gaussians in the graph.
     * @param gi A reference to the graph interface containing Gaussian data.
     */
    template<typename T>
    inline LaplacianMatrices BuildGmmApproxLaplacian(GraphInterface &gi, T iso_value = 0.01) {
        Stopwatch total_stopwatch("Total BuildGmmApproxLaplacian");

        LaplacianMatrices matrices;
        auto mu_ij = gi.edge_property<Vector<T, 3> >("e:mu_ij", Vector<T, 3>::Zero());
        auto proj_m_ij = gi.edge_property<Vector<T, 3> >("e:proj_mu_ij", Vector<T, 3>::Zero());
        auto sigma_ij = gi.edge_property<Matrix<T, 3, 3> >("e:sigma_ij", Matrix<T, 3, 3>::Zero());
        auto normal_ij = gi.edge_property<Vector<T, 3> >("e:normal_ij", Vector<T, 3>::Zero());
        auto a_ij = gi.edge_property<T>("e:a_ij", 0);
        auto m_ij = gi.edge_property<T>("e:m_ij", 0);
        auto k_ij = gi.edge_property<T>("e:k_ij", 0);
        auto trace_ij = gi.edge_property<T>("e:trace_ij", 0);

        auto m_i = gi.vertex_property<Vector<T, 3> >("v:mu_i", Vector<T, 3>::Zero());
        auto scale = gi.vertex_property<Vector<T, 3> >("v:scale", Vector<T, 3>(0.0f));
        auto rotation = gi.vertex_property<Vector<T, 4> >("v:rotation", Vector<T, 4>(0.0f));
        auto sigma_i = gi.vertex_property<Matrix<T, 3, 3> >("v:sigma_i", Matrix<T, 3, 3>::Identity());
        auto sigma_i_inverse = gi.vertex_property<Matrix<T, 3, 3> >("v:sigma_i_inverse", Matrix<T, 3, 3>::Identity());
        auto a_ii = gi.vertex_property<T>("v:stiffness_diag");
        auto m_ii = gi.vertex_property<T>("v:mass_diag"); {
            Stopwatch sw("Stage 1: Precomputing per-vertex covariances");
            for (const auto v_i: gi.vertices) {
                m_i[v_i] = MapConst(glm::dvec3(gi.vpoint[v_i]));
                auto v_i_quat = Eigen::Quaternion<T>(rotation[v_i].w, rotation[v_i].x, rotation[v_i].y,
                                                     rotation[v_i].z);
                auto v_i_scale = Vector<T, 3>(scale[v_i].x, scale[v_i].y, scale[v_i].z);
                CovarianceInterface<T> i_cov_i(v_i_scale, v_i_quat);
                sigma_i[v_i] = i_cov_i.get_covariance_matrix();
                sigma_i_inverse[v_i] = sigma_i[v_i].inverse();
            }
        } {
            Stopwatch sw("Stage 2: Precomputing per-edge product Gaussians");
            for (const auto edge: gi.edges) {
                auto h0 = gi.get_halfedge(edge, 0);
                auto v_i = gi.from_vertex(h0);
                auto v_j = gi.to_vertex(h0);

                const Vector<T, 3> mu_i = m_i[v_i];
                const Vector<T, 3> mu_j = m_i[v_j];

                const Matrix<T, 3, 3> &Sigma_i = sigma_i[v_i];
                const Matrix<T, 3, 3> &Sigma_j = sigma_i[v_j];

                const Matrix<T, 3, 3> &Sigma_i_inv = sigma_i_inverse[v_i];
                const Matrix<T, 3, 3> &Sigma_j_inv = sigma_i_inverse[v_j];

                mu_ij[edge] = compute_mu_ij(mu_i, Sigma_i, mu_j, Sigma_j);;
                sigma_ij[edge] = (Sigma_i_inv + Sigma_j_inv).inverse();
            }
        }

        std::vector<Eigen::Triplet<T> > triplets_A;
        std::vector<Eigen::Triplet<T> > triplets_M; {
            Stopwatch sw("Stage 3: Per-edge projection and term evaluation");
            for (const auto edge: gi.edges) {
                auto h0 = gi.get_halfedge(edge, 0);
                auto v_i = gi.from_vertex(h0);
                auto v_j = gi.to_vertex(h0);

                const Vector<T, 3> mu_i = MapConst(glm::dvec3(gi.vpoint[v_i]));
                const Vector<T, 3> mu_j = MapConst(glm::dvec3(gi.vpoint[v_j]));

                const Vector<T, 3> &Mu_ij = mu_ij[edge];
                const Matrix<T, 3, 3> &Sigma_ij = sigma_ij[edge];

                const Matrix<T, 3, 3> &Sigma_i = sigma_i[v_i];
                const Matrix<T, 3, 3> &Sigma_j = sigma_i[v_j];

                proj_m_ij[edge] = project_point_to_surface(mu_ij[edge], m_i.vector(), sigma_i.vector(), {}, iso_value);
                normal_ij[edge] = compute_surface_normal(proj_m_ij[edge], m_i.vector(), sigma_i.vector(), {});

                // 2. Compute the geometric setup at hat_mu_ij
                Matrix<T, 3, 3> P = compute_projection_matrix_P(normal_ij[edge]);
                Eigen::Matrix<T, 3, 2> V = compute_tangent_basis(normal_ij[edge]);

                // 3. Compute the 2D projected covariances
                Matrix<T, 2, 2> tilde_Sigma_i = compute_tilde_Sigma(Sigma_i, V);
                Matrix<T, 2, 2> tilde_Sigma_j = compute_tilde_Sigma(Sigma_j, V);
                Matrix<T, 2, 2> tilde_Sigma_ij = compute_tilde_Sigma(Sigma_ij, V);

                // 4. Calculate the Galerkin matrix components
                T K_ij = compute_K_ij(mu_i, Sigma_i, mu_j, Sigma_j);
                T M_ij = compute_M_ij_approx(K_ij, tilde_Sigma_ij);

                T trace_term = compute_trace_term_A_ij(tilde_Sigma_i, tilde_Sigma_j, tilde_Sigma_ij);
                T projection_term = compute_projection_term_A_ij(mu_i, Sigma_i, mu_j, Sigma_j, Mu_ij, P);

                // The full approximation for A_ij would be M_ij * (trace_term + projection_term)
                T A_ij = M_ij * (trace_term + projection_term);

                m_ij[edge] = A_ij;
                a_ij[edge] = M_ij;
                k_ij[edge] = K_ij;
                trace_ij[edge] = trace_term;

                T stiffness_val = a_ij[edge];
                T mass_val = m_ij[edge];

                auto v_i_idx = v_i.idx();
                auto v_j_idx = v_j.idx();

                // Add the symmetric off-diagonal entries
                triplets_A.emplace_back(v_i_idx, v_j_idx, stiffness_val);
                triplets_A.emplace_back(v_j_idx, v_i_idx, stiffness_val);

                triplets_M.emplace_back(v_j_idx, v_i_idx, mass_val);
                triplets_M.emplace_back(v_i_idx, v_j_idx, mass_val);

                // Accumulate the sums for the diagonal entries
                a_ii[v_i] -= stiffness_val;
                a_ii[v_j] -= stiffness_val;

                // For the lumped mass matrix, the diagonal is the sum of off-diagonals
                m_ii[v_i] += mass_val;
                m_ii[v_j] += mass_val;
            }
        } {
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
