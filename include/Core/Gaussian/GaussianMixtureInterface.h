#pragma once

#include "AABBUtils.h"
#include "PointCloudInterface.h"
#include "Logger.h"
#include "Octree.h"
#include "LaplacianOperator.h"
#include "MatTraits.h"
#include "GraphInterface.h"
#include "GmmUtils.h"

#include <vector>
#include <chrono>


namespace Bcg {
    class GaussianMixtureInterface : public PointCloudInterface {
    public:
        VertexProperty<PointType> means;
        VertexProperty<Vector<float, 3> > scale;
        VertexProperty<Vector<float, 4> > rotation;
        VertexProperty<Matrix<float, 3, 3> > covariances;
        VertexProperty<Matrix<float, 3, 3> > covariances_inv;
        VertexProperty<float> weights;
        VertexProperty<float> norm_factors;
        Octree octree;

        explicit GaussianMixtureInterface(PointCloudData &data) : PointCloudInterface(data.vertices) {
        }

        explicit GaussianMixtureInterface(Vertices &vertices) : PointCloudInterface(vertices),
                                                                means(vertices.vertex_property<PointType>("v:point")),
                                                                covariances(
                                                                    vertices.vertex_property<Matrix<float, 3, 3> >(
                                                                        "v:covs")),
                                                                covariances_inv(
                                                                    vertices.vertex_property<Matrix<float, 3, 3> >(
                                                                        "v:covs_inv")),
                                                                weights(vertices.vertex_property<float>(
                                                                    "v:weights")),
                                                                norm_factors(
                                                                    vertices.vertex_property<float>("v:norm_factors")) {
            assert(norm_factors && norm_factors.name() == "v:norm_factors");
            assert(means && means.name() == "v:point");
            assert(covariances && covariances.name() == "v:covs");
            assert(covariances_inv && covariances_inv.name() == "v:covs_inv");
            assert(weights && weights.name() == "v:weights");
        }

        GaussianMixtureInterface() = default;

        Property<AABB<float> > ConvertGaussiansToAABBs() {
            auto aabbs = vertices.vertex_property<AABB<float> >("v:aabb");
            aabbs.vector() = compute_aabbs_from(means.vector(), covariances.vector());

            for (const auto v: vertices) {
                aabbs[v] = constructAABBForGaussian(means[v], covariances[v], k_sigma, min_half_extent);
            }
            return aabbs;
        }

        void build() {
            if (!scale) {
                scale = vertices.vertex_property<Vector<float, 3> >("v:scale");
            }
            if (!rotation) {
                rotation = vertices.vertex_property<Vector<float, 4> >("v:rotation");
            }

            if (scale && rotation) {
                set_covs(scale.vector(), rotation.vector());
            } else {
                Log::Error("Missing 'v:scale' or 'v:rotation' vertex properties.");
                return;
            }
            octree = Octree();
            auto aabbs = ConvertGaussiansToAABBs(3.0f, 0.0f);
            // Build the octree with a maximum of 32 elements per node and a maximum depth
            octree.build(aabbs, {Octree::SplitPoint::Median, true, 0.0f}, 32, 10);
        }

        void set_covs(const std::vector<Matrix<float, 3, 3> > &covs) {
            if (covs.size() != vertices.size()) {
                Log::Error("Size of scaling does not match Size of vertices");
                return;
            }

            for (const auto v: vertices) {
                // Precompute inverse covariance matrix
                covariances[v] = covs[v.idx()];
            }

            compute_covs_inverse();
            compute_norm_factors();
        }

        void compute_covs_inverse() {
            for (const auto v: vertices) {
                covariances_inv[v] = glm::inverse(covariances[v]);
            }
        }

        void compute_norm_factors() {
            if (!norm_factors) {
                norm_factors = vertices.vertex_property<float>("v:norm_factors");
            }

            for (const auto v: vertices) {
                // Precompute the normalization factor for the PDF
                float det = glm::determinant(covariances[v]);
                if (det > 1e-9) {
                    // Avoid division by zero or negative sqrt
                    float two_pi_cubed = std::pow(2.0f * glm::pi<float>(), 3.0f);
                    norm_factors[v] = 1.0f / std::sqrt(two_pi_cubed * det);
                } else {
                    norm_factors[v] = 0.0f; // This Gaussian is degenerate and will have zero probability
                }
            }
        }


        void set_covs(const std::vector<Vector<float, 3> > &scaling,
                      const std::vector<Vector<float, 4> > &quaternions) {
            if (scaling.size() != quaternions.size()) {
                Log::Error("Size of scaling does not match size of quaternions");
                return;
            }
            if (scaling.size() != vertices.size()) {
                Log::Error("Size of scaling does not match size of vertices");
                return;
            }

            covariances = vertices.vertex_property<Matrix<float,3,3>>("v:covs");

            for (const auto v : vertices) {
                const size_t i = v.idx();

                // 1) Build & normalize quaternion (w,x,y,z)
                glm::quat q(quaternions[i].w, quaternions[i].x, quaternions[i].y, quaternions[i].z);
                q = glm::normalize(q);

                // 2) Rotation in GLM, then convert once
                Matrix<float,3,3> R = glm::mat3_cast(q);     // from your GlmToEigen.h (or equivalent)

                // 3) Diagonal of variances (σ^2). Clamp to epsilon to avoid singular Σ.
                const float sx = std::max(scaling[i].x, 1e-8f);
                const float sy = std::max(scaling[i].y, 1e-8f);
                const float sz = std::max(scaling[i].z, 1e-8f);

                Matrix<float,3,3> D(0.0f);
                D[0][0] = sx * sx;
                D[1][1] = sy * sy;
                D[2][2] = sz * sz;

                // 4) Σ = R D R^T (use your Matrix ops, not glm::transpose on a non-glm type)
                Matrix<float,3,3> Rt = glm::transpose(R);
                covariances[v] = R * D * Rt;
            }

            compute_covs_inverse();
            compute_norm_factors();
        }

        std::vector<float> pdf(const std::vector<Vector<float, 3> > &query_points) const {
            std::vector<float> results;
            results.reserve(query_points.size());

            // Temporary vector to store results from the octree query.
            // Re-using its memory avoids re-allocation in every loop iteration.
            std::vector<size_t> candidate_indices;

            for (const auto &query_point: query_points) {
                // --- 1. BROAD PHASE ---
                // Query the octree to find all Gaussians whose AABBs contain the query point.
                // A point is a degenerate AABB where min == max.
                AABB<float> point_aabb(query_point, query_point);
                octree.query(point_aabb, candidate_indices);

                // --- 2. NARROW PHASE ---
                // Sum the PDFs of the candidate Gaussians. Use double for precision during summation.
                double total_pdf = 0.0;
                for (const size_t idx: candidate_indices) {
                    Vertex v(idx);

                    // Skip degenerate Gaussians
                    if (norm_factors[v] == 0.0f) continue;

                    // Calculate the Mahalanobis distance squared: (x-μ)ᵀ * Σ⁻¹ * (x-μ)
                    Vector<float, 3> delta = query_point - means[v];
                    float mahalanobis_sq = glm::dot(delta, covariances_inv[v] * delta);

                    // Calculate the PDF for this single Gaussian
                    double single_pdf = norm_factors[v] * std::exp(-0.5f * mahalanobis_sq);

                    // Add the weighted PDF to the total
                    total_pdf += weights[v] * single_pdf;
                }
                results.push_back(static_cast<float>(total_pdf));
            }
            return results;
        }

        std::vector<Vector<float, 3> > gradient(const std::vector<Vector<float, 3> > &query_points) const {
            std::vector<Vector<float, 3> > results;
            results.reserve(query_points.size());

            std::vector<size_t> candidate_indices;

            for (const auto &query_point: query_points) {
                // --- 1. BROAD PHASE (Identical to pdf method) ---
                AABB<float> point_aabb(query_point, query_point);
                octree.query(point_aabb, candidate_indices);

                // --- 2. NARROW PHASE ---
                Vector<double, 3> total_gradient(0.0, 0.0, 0.0); // Use double for accumulation
                for (const size_t idx: candidate_indices) {
                    Vertex v(idx);

                    if (norm_factors[v] == 0.0f) continue;

                    Vector<float, 3> delta = query_point - means[v];
                    float mahalanobis_sq = glm::dot(delta, covariances_inv[v] * delta);

                    // First, calculate the PDF for this Gaussian (N(x))
                    float single_pdf = norm_factors[v] * std::exp(-0.5f * mahalanobis_sq);

                    // Now, calculate the gradient for this Gaussian: -Σ⁻¹(x-μ) * N(x)
                    Vector<float, 3> single_gradient = -1.0f * (covariances_inv[v] * delta) * single_pdf;

                    // Add the weighted gradient to the total
                    total_gradient += weights[v] * single_gradient;
                }
                results.push_back(Vector<float, 3>(total_gradient));
            }
            return results;
        }

        struct EvalPoint {
            Vector<float, 3> point;
            Vector<float, 3> gradient = Vector<float, 3>(0.0f, 0.0f, 0.0f);
            float pdf = 0.0;

            friend std::ostream &operator<<(std::ostream &os, const EvalPoint &ep) {
                os << "Point: (" << ep.point.x << ", " << ep.point.y << ", " << ep.point.z << "), "
                   << "PDF: " << ep.pdf << ", "
                   << "Gradient: (" << ep.gradient.x << ", " << ep.gradient.y << ", " << ep.gradient.z << ")";
                return os;
            }
        };

        void pdf_and_gradient(const Vector<float, 3> &query_point, Vector<float, 3> &total_gradient, float &total_pdf) {
            std::vector<size_t> candidate_indices;
            AABB<float> point_aabb(query_point, query_point);
            octree.query(point_aabb, candidate_indices);

            total_pdf = 0.0;
            total_gradient = Vector<float, 3>(0.0, 0.0, 0.0); // Use double for accumulation
            for (const size_t idx: candidate_indices) {
                Vertex v(idx);

                if (norm_factors[v] == 0.0f) continue;

                Vector<float, 3> delta = query_point - means[v];
                float mahalanobis_sq = glm::dot(delta, covariances_inv[v] * delta);

                // First, calculate the PDF for this Gaussian (N(x))
                float single_pdf = norm_factors[v] * std::exp(-0.5f * mahalanobis_sq);

                total_pdf += weights[v] * single_pdf;

                // Now, calculate the gradient for this Gaussian: -Σ⁻¹(x-μ) * N(x)
                Vector<float, 3> single_gradient = -1.0f * (covariances_inv[v] * delta) * single_pdf;

                // Add the weighted gradient to the total
                total_gradient += weights[v] * single_gradient;
            }
        }

        void pdfs_and_gradients(const std::vector<Vector<float, 3> > &points,
                                std::vector<Vector<float, 3> > &gradients,
                                std::vector<float> &pdfs) const {
            std::vector<size_t> candidate_indices;

            for (size_t i = 0; i < points.size(); ++i) {
                const auto &query_point = points[i];
                // --- 1. BROAD PHASE (Identical to pdf method) ---
                AABB<float> point_aabb(query_point, query_point);
                octree.query(point_aabb, candidate_indices);

                // --- 2. NARROW PHASE ---
                float total_pdf = 0.0;
                Vector<double, 3> total_gradient(0.0, 0.0, 0.0); // Use double for accumulation
                for (const size_t idx: candidate_indices) {
                    Vertex v(idx);

                    if (norm_factors[v] == 0.0f) continue;

                    Vector<float, 3> delta = query_point - means[v];
                    float mahalanobis_sq = glm::dot(delta, covariances_inv[v] * delta);

                    // First, calculate the PDF for this Gaussian (N(x))
                    float single_pdf = norm_factors[v] * std::exp(-0.5f * mahalanobis_sq);

                    total_pdf += weights[v] * single_pdf;

                    // Now, calculate the gradient for this Gaussian: -Σ⁻¹(x-μ) * N(x)
                    Vector<float, 3> single_gradient = -1.0f * (covariances_inv[v] * delta) * single_pdf;

                    // Add the weighted gradient to the total
                    total_gradient += weights[v] * single_gradient;
                }
                gradients[i] = total_gradient;
                pdfs[i] = total_pdf;
            }
        }

        void project_point_to_surface(std::vector<Vector<float, 3> > &points,
                                      std::vector<Vector<float, 3> > &gradients,
                                      std::vector<float> &pdfs,
                                      float iso_value,
                                      int iterations = 5) {
            std::vector<size_t> active_indices(points.size());
            std::iota(active_indices.begin(), active_indices.end(), 0);

            for (int i = 0; i < iterations; ++i) {
                // 1. Evaluate the field value F(x_k) and 2. Evaluate the gradient grad F(x_k)
                pdfs_and_gradients(points, gradients, pdfs);

                for (size_t active_idx = 0; active_idx < active_indices.size(); /* no increment here */) {
                    size_t point_idx = active_indices[active_idx];

                    const Vector<float, 3> &grad_F_x = gradients[point_idx];
                    float grad_norm_sq = VecTraits<Vector<float, 3> >::squared_length(grad_F_x);

                    if (grad_norm_sq < 1e-12) {
                        // Remove this point by swapping with the last and popping.
                        // Do not increment active_idx, as we need to process the new element at this position.
                        active_indices[active_idx] = active_indices.back();
                        active_indices.pop_back();
                    } else {
                        const float F_x = pdfs[point_idx];
                        float error = F_x - iso_value;
                        points[point_idx] -= (error / grad_norm_sq) * grad_F_x;

                        // Only increment if we didn't remove the element.
                        ++active_idx;
                    }
                }
            }
        }

        template<typename T>
        Vector<T, 3> compute_mu_ij(const Vector<T, 3> &mu_i, const Matrix<T, 3, 3> &Sigma_i,
                                   const Vector<T, 3> &mu_j, const Matrix<T, 3, 3> &Sigma_j) {
            Matrix<T, 3, 3> Sigma_i_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_i);
            Matrix<T, 3, 3> Sigma_j_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_j);

            Matrix<T, 3, 3> Sigma_ij = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_i_inv + Sigma_j_inv);

            return Sigma_ij * (Sigma_i_inv * mu_i + Sigma_j_inv * mu_j);
        }

        template<typename T>
        Matrix<T, 3, 3> compute_projection_matrix_P(const Vector<T, 3> &normal) {
            // Assumes `normal` is already normalized for efficiency.
            return MatTraits<Matrix<T, 3, 3> >::identity() - VecTraits<Vector<T, 3> >::outer_product(normal, normal);
        }

        template<typename T>
        Matrix<T, 2, 3> compute_tangent_basis(const Vector<T, 3> &normal) {
            Vector<T, 3> a = (std::abs(normal.x) > T(0.9)) ? Vector<T, 3>(0, 1, 0) : Vector<T, 3>(1, 0, 0);
            Vector<T, 3> v1 = VecTraits<Vector<T, 3> >::normalize(VecTraits<Vector<T, 3> >::cross(normal, a));
            Vector<T, 3> v2 = VecTraits<Vector<T, 3> >::normalize(VecTraits<Vector<T, 3> >::cross(normal, v1));

            // GLM mat<C,R>: C columns, R rows -> Matrix<T, 2, 3> is 3 rows x 2 cols
            Matrix<T, 2, 3> V;
            V[0] = v1; // column 0
            V[1] = v2; // column 1
            return V;
        }

        template<typename T>
        Matrix<T, 2, 2> compute_tilde_Sigma(const Matrix<T, 3, 3> &Sigma,
                                            const Matrix<T, 2, 3> &tangent_basis_V) {
            return MatTraits<Matrix<T, 3, 2> >::transpose(tangent_basis_V) * Sigma * tangent_basis_V;
        }

        template<typename T>
        T compute_K_ij(const Vector<T, 3> &mu_i, const Matrix<T, 3, 3> &Sigma_i,
                       const Vector<T, 3> &mu_j, const Matrix<T, 3, 3> &Sigma_j) {
            Vector<T, 3> mu_diff = mu_i - mu_j;
            Matrix<T, 3, 3> Sigma_sum_inv = MatTraits<Matrix<T, 3, 3> >::inverse(Sigma_i + Sigma_j);

            T exponent = -0.5 * VecTraits<Vector<float, 3> >::dot(mu_diff, Sigma_sum_inv * mu_diff);

            return std::exp(exponent);
        }

        template<typename T>
        T compute_M_ij_approx(T K_ij, const Matrix<T, 2, 2> &tilde_Sigma_ij) {
            T det = MatTraits<Matrix<T, 2, 2> >::determinant(tilde_Sigma_ij);
            if (det <= 0) {
                // Handle potential numerical issues with non-positive definite matrices
                return 0.0;
            }
            return K_ij * 2.0 * M_PI * std::sqrt(det);
        }

        template<typename T>
        T compute_trace_term_A_ij(const Matrix<T, 2, 2> &tilde_Sigma_i, const Matrix<T, 2, 2> &tilde_Sigma_j,
                                  const Matrix<T, 2, 2> &tilde_Sigma_ij) {
            Matrix<T, 2, 2> product = MatTraits<Matrix<T, 2, 2> >::inverse(tilde_Sigma_j) * MatTraits<Matrix<T, 2,
                                          2> >::inverse(tilde_Sigma_i) * tilde_Sigma_ij;
            return product[0][0] + product[1][1];
        }

        template<typename T>
        T compute_projection_term_A_ij(
            const Vector<T, 3> &mu_i, const Matrix<T, 3, 3> &Sigma_i_inv,
            const Vector<T, 3> &mu_j, const Matrix<T, 3, 3> &Sigma_j_inv,
            const Vector<T, 3> &mu_ij, const Matrix<T, 3, 3> &P) {
            Vector<T, 3> vec_i = Sigma_i_inv * (mu_ij - mu_i);
            Vector<T, 3> vec_j = Sigma_j_inv * (mu_ij - mu_j);

            // The result is a 1x1 matrix, so we extract the scalar value.
            return VecTraits<Vector<float, 3> >::dot(vec_j, P * vec_i);
        }

        void compute_mus_and_sigmas(GraphInterface &gi) {
            auto sigma_ij = gi.edges.edge_property<Matrix<float, 3, 3> >("e:sigma_ij", Matrix<float, 3, 3>(1.0));
            auto mu_ij = gi.edges.edge_property<Vector<float, 3> >("e:mu_ij", Vector<float, 3>(0.0));
            auto eval_points = gi.edges.edge_property<EvalPoint>("e:eval_points");
            Stopwatch sw("Stage 1: Compute mu_ij and sigma_ij");
            for (const auto edge: gi.edges) {
                auto h0 = gi.get_halfedge(edge, 0);
                auto v_i = gi.from_vertex(h0);
                auto v_j = gi.to_vertex(h0);

                const Vector<float, 3> mu_i = means[v_i];
                const Vector<float, 3> mu_j = means[v_j];


                const Matrix<float, 3, 3> &Sigma_i_inv = covariances_inv[v_i];
                const Matrix<float, 3, 3> &Sigma_j_inv = covariances_inv[v_j];

                sigma_ij[edge] = MatTraits<Matrix<float, 3, 3> >::inverse(Sigma_i_inv + Sigma_j_inv);
                mu_ij[edge] = sigma_ij[edge] * (Sigma_i_inv * mu_i + Sigma_j_inv * mu_j);
                eval_points[edge].point = mu_ij[edge];
            }
        }

        void initialize_eval_points(std::vector<Vector<float, 3> > &points,
                                    std::vector<Vector<float, 3> > &gradients,
                                    std::vector<float> &pdfs) {
            Stopwatch sw("Stage 2: Per-edge projection and term evaluation");
            pdfs_and_gradients(points, gradients, pdfs);
        }

        LaplacianMatrices compute_laplacian_matrices(Halfedges &halfedges, Edges &edges) {
            auto mu_ij = edges.edge_property<Vector<float, 3> >("e:mu_ij", Vector<float, 3>(0.0));
            auto proj_m_ij = edges.edge_property<Vector<float, 3> >("e:proj_mu_ij", Vector<float, 3>(0.0));
            auto sigma_ij = edges.edge_property<Matrix<float, 3, 3> >("e:sigma_ij", Matrix<float, 3, 3>(1.0));
            auto normal_ij = edges.edge_property<Vector<float, 3> >("e:normal_ij", Vector<float, 3>(0.0));
            auto a_ij = edges.edge_property<float>("e:a_ij", 0);
            auto m_ij = edges.edge_property<float>("e:m_ij", 0);
            auto k_ij = edges.edge_property<float>("e:k_ij", 0);
            auto trace_ij = edges.edge_property<float>("e:trace_ij", 0);
            auto pdf_ij = edges.edge_property<float>("e:pdf_ij", 0);
            auto grad_ij = edges.edge_property<Vector<float, 3> >("e:grad_ij", Vector<float, 3>(0.0));

            auto a_ii = vertices.vertex_property<float>("v:stiffness_diag");
            auto m_ii = vertices.vertex_property<float>("v:mass_diag");

            compute_covs_inverse();
            auto gi = GraphInterface(vertices, halfedges, edges);

            compute_mus_and_sigmas(gi);
            proj_m_ij.vector() = mu_ij.vector();
            initialize_eval_points(proj_m_ij.vector(), grad_ij.vector(), pdf_ij.vector());


            std::vector<Eigen::Triplet<float> > triplets_A;
            std::vector<Eigen::Triplet<float> > triplets_M; {
                Stopwatch sw("Stage 3: Project to surface");
                project_point_to_surface(proj_m_ij.vector(), grad_ij.vector(), pdf_ij.vector(), 0.01f, 5);
            } {
                Stopwatch sw("Stage 4: Compute Laplacian terms and assemble matrices");

                for (const auto edge: edges) {
                    auto h0 = gi.get_halfedge(edge, 0);
                    auto v_i = gi.from_vertex(h0);
                    auto v_j = gi.to_vertex(h0);

                    normal_ij[edge] = VecTraits<Vector<float, 3> >::normalize(-grad_ij[edge]);

                    Matrix<float, 3, 3> P = compute_projection_matrix_P(normal_ij[edge]);
                    Matrix<float, 2, 3> V = compute_tangent_basis(normal_ij[edge]);

                    const Matrix<float, 3, 3> &Sigma_i = covariances[v_i];
                    const Matrix<float, 3, 3> &Sigma_j = covariances[v_j];
                    const Matrix<float, 3, 3> &Sigma_ij = sigma_ij[edge];

                    // 3. Compute the 2D projected covariances
                    Matrix<float, 2, 2> tilde_Sigma_i = compute_tilde_Sigma(Sigma_i, V);
                    Matrix<float, 2, 2> tilde_Sigma_j = compute_tilde_Sigma(Sigma_j, V);
                    Matrix<float, 2, 2> tilde_Sigma_ij = compute_tilde_Sigma(Sigma_ij, V);

                    const Vector<float, 3> &mu_i = means[v_i];
                    const Vector<float, 3> &mu_j = means[v_j];
                    const Vector<float, 3> &Mu_ij = mu_ij[edge];

                    // 4. Calculate the Galerkin matrix components
                    const float K_ij = compute_K_ij(mu_i, Sigma_i, mu_j, Sigma_j);
                    const float M_ij = compute_M_ij_approx(K_ij, tilde_Sigma_ij);
                    const float tr_ij = compute_trace_term_A_ij(tilde_Sigma_i, tilde_Sigma_j, tilde_Sigma_ij);
                    const float pr_ij = compute_projection_term_A_ij(mu_i, Sigma_i, mu_j, Sigma_j, Mu_ij, P);


                    // The full approximation for A_ij would be M_ij * (trace_term + projection_term)
                    const float A_ij = M_ij * (tr_ij + pr_ij); // stiffness contribution
                    a_ij[edge] = A_ij;
                    m_ij[edge] = M_ij;
                    k_ij[edge] = K_ij;
                    trace_ij[edge] = tr_ij;

                    float stiffness_val = a_ij[edge];
                    float mass_val = m_ij[edge];

                    auto v_i_idx = v_i.idx();
                    auto v_j_idx = v_j.idx();

                    // Add the symmetric off-diagonal entries
                    triplets_A.emplace_back(v_i_idx, v_j_idx, -stiffness_val);
                    triplets_A.emplace_back(v_j_idx, v_i_idx, -stiffness_val);

                    triplets_M.emplace_back(v_j_idx, v_j_idx, mass_val);
                    triplets_M.emplace_back(v_j_idx, v_j_idx, mass_val);

                    // Accumulate the sums for the diagonal entries
                    a_ii[v_i] += stiffness_val;
                    a_ii[v_j] += stiffness_val;

                    // For the lumped mass matrix, the diagonal is the sum of off-diagonals
                    m_ii[v_i] += mass_val;
                    m_ii[v_j] += mass_val;
                }
            }

            for (const auto v_i: gi.vertices) {
                const auto i = v_i.idx();
                triplets_A.emplace_back(i, i, a_ii[v_i]);

                // For the mass matrix, either use the lumped version (diag_M)
                // or compute a more accurate diagonal term if you have it.
                // Let's stick with the lumped mass for now.
                // M_ii = integral of phi_i^2. A simple approx is 1.0 or sum of row.
                // Let's add the diagonal sum plus a base value (e.g., 1.0) to keep it well-conditioned.
                // A proper M_ii would require projecting and integrating phi_i^2.
                const float mdiag = std::max(m_ii[v_i], 1e-9f);
                triplets_M.emplace_back(i, i, mdiag);
            }
            LaplacianMatrices matrices;
            matrices.build(triplets_A, triplets_M, gi.vertices.n_vertices());
            return matrices;
        }
    };
}
