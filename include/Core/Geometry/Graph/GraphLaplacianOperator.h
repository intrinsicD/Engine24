//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_SURFACEMESHLAPLACIANOPERATOR_H
#define ENGINE24_SURFACEMESHLAPLACIANOPERATOR_H

#include "LaplacianOperator.h"
#include "Graph.h" // Assuming this provides vertex, edge, face iterators and connectivity
#include "Eigen/Geometry"
#include "CovarianceInterface.h"
#include "GlmToEigen.h"
#include "GaussianMixtureInterface.h"

#include <vector>

namespace Bcg {
    inline LaplacianMatrices ComputeHeatKernelLaplacianOperator(GraphInterface &graph, double t_factor = 1.0) {
        LaplacianMatrices matrices;
        const long n_vertices = graph.vertices.n_vertices();
        const long n_edges = graph.edges.n_edges();
        if (n_vertices == 0) return matrices;

        // --- Step 1: Compute all edge lengths in a single pass ---
        std::vector<double> edge_lengths;
        edge_lengths.reserve(n_edges);
        double total_edge_length = 0.0;

        for (const auto edge: graph.edges) {
            auto h0 = graph.get_halfedge(edge, 0);
            auto v_i = graph.from_vertex(h0);
            auto v_j = graph.to_vertex(h0);
            const PointType &p_i = graph.vpoint[v_i];
            const PointType &p_j = graph.vpoint[v_j];

            const double length = glm::distance(glm::dvec3(p_i), glm::dvec3(p_j));
            edge_lengths.push_back(length);
            total_edge_length += length;
        }

        // --- Step 2: Compute the robust, data-driven scale 't' ---
        double t_scaled = 1.0; // Default value in case of no edges
        if (n_edges > 0) {
            const double avg_edge_length = total_edge_length / n_edges;
            t_scaled = t_factor * avg_edge_length * avg_edge_length;
        }

        // --- Step 3: Build the triplets using the pre-computed lengths ---
        std::vector<Eigen::Triplet<float> > S_triplets;
        Eigen::VectorXd M_diagonal_values = Eigen::VectorXd::Zero(n_vertices);

        for (long i = 0; i < n_edges; ++i) {
            const auto edge = Edge(i);
            const double distance = edge_lengths[i];

            // w_ij = exp(-||p_i - p_j||^2 / (4*t))
            const double weight = std::exp(-distance * distance / (4.0 * t_scaled));

            auto h0 = graph.get_halfedge(edge, 0);
            auto v_i_idx = graph.from_vertex(h0).idx();
            auto v_j_idx = graph.to_vertex(h0).idx();

            S_triplets.emplace_back(v_i_idx, v_j_idx, -weight);
            S_triplets.emplace_back(v_j_idx, v_i_idx, -weight);


            M_diagonal_values[v_i_idx] += weight;
            M_diagonal_values[v_j_idx] += weight;
        }

        for (long i = 0; i < n_vertices; ++i) {
            S_triplets.emplace_back(i, i, M_diagonal_values[i]);
        }

        // --- Step 4: Finalize the Mass Matrix ---
        std::vector<Eigen::Triplet<float> > M_triplets;
        M_triplets.reserve(n_vertices);
        for (long i = 0; i < n_vertices; ++i) {
            if (M_diagonal_values[i] > 1e-9) {
                M_triplets.emplace_back(i, i, M_diagonal_values[i]);
            } else {
                M_triplets.emplace_back(i, i, 1e-9);
            }
        }

        matrices.build(S_triplets, M_triplets, n_vertices);
        return matrices;
    }

    inline LaplacianMatrices ComputeGraphLaplacianOperator(GraphInterface &graph) {
        LaplacianMatrices matrices;
        const long n_vertices = graph.vertices.n_vertices();
        if (n_vertices == 0) return matrices;

        std::vector<Eigen::Triplet<float> > S_triplets;
        // Reserve space: 2 triplets for each edge (i,j) and (j,i), plus n for the diagonal.
        S_triplets.reserve(graph.edges.n_edges() * 2 + n_vertices);

        Eigen::VectorXf diagonal_degrees = Eigen::VectorXf::Zero(n_vertices);

        // Iterate over all edges to set off-diagonal entries
        for (const auto edge: graph.edges) {
            auto h0 = graph.get_halfedge(edge, 0);
            auto v_i = graph.from_vertex(h0);
            auto v_j = graph.to_vertex(h0);

            // Off-diagonal elements are -1 for every connected edge
            S_triplets.emplace_back(v_i.idx(), v_j.idx(), -1.0f);
            S_triplets.emplace_back(v_j.idx(), v_i.idx(), -1.0f);

            // Accumulate the degree for the diagonal
            diagonal_degrees[v_i.idx()]++;
            diagonal_degrees[v_j.idx()]++;
        }

        // Add the diagonal entries (the vertex degrees)
        for (long i = 0; i < n_vertices; ++i) {
            S_triplets.emplace_back(i, i, diagonal_degrees[i]);
        }

        matrices.build(S_triplets, {}, n_vertices);
        matrices.M.resize(n_vertices, n_vertices);
        matrices.M.setIdentity();

        return matrices;
    }



    inline LaplacianMatrices ComputeGMMLaplacianOperator(GraphInterface &graph, double t_factor = 1.0) {
        GaussianMixtureInterface gmm(graph.vertices);
        gmm.build();
        return gmm.compute_laplacian_matrices(graph.halfedges, graph.edges);
    }

    inline LaplacianMatrices ComputeGMMGraphLaplacianOperator(GraphInterface &graph, double t_factor = 1.0) {
        LaplacianMatrices matrices;
        const long n_vertices = graph.vertices.n_vertices();
        if (n_vertices == 0) return matrices;

        std::vector<Eigen::Triplet<float> > S_triplets;
        // Reserve space: 2 triplets for each edge (i,j) and (j,i), plus n for the diagonal.
        S_triplets.reserve(graph.edges.n_edges() * 2 + n_vertices);

        Eigen::VectorXd M_diagonal_values = Eigen::VectorXd::Zero(n_vertices);

        auto scale = graph.vertex_property<Vector<float, 3> >("v:scale", Vector<float, 3>(0.0f));
        auto rotation = graph.vertex_property<Vector<float, 4> >("v:rotation", Vector<float, 4>(0.0f));

        if (!scale || !rotation) {
            Log::Error("ComputeGMMLaplacianOperator: Missing 'v:scale' or 'v:rotation' vertex properties.");
            return matrices;
        }

        // Iterate over all edges to set off-diagonal entries
        for (const auto edge: graph.edges) {
            auto h0 = graph.get_halfedge(edge, 0);
            auto v_i = graph.from_vertex(h0);
            auto v_j = graph.to_vertex(h0);

            auto v_i_quat = Eigen::Quaternion<double>(rotation[v_i].w, rotation[v_i].x, rotation[v_i].y,
                                                      rotation[v_i].z);
            auto v_i_scale = Eigen::Vector3d(scale[v_i].x, scale[v_i].y, scale[v_i].z);
            CovarianceInterface<double> i_cov_i(v_i_scale, v_i_quat);

            auto v_j_quat = Eigen::Quaternion<double>(rotation[v_j].w, rotation[v_j].x, rotation[v_j].y,
                                                      rotation[v_j].z);
            auto v_j_scale = Eigen::Vector3d(scale[v_j].x, scale[v_j].y, scale[v_j].z);
            CovarianceInterface<double> i_cov_j(v_j_scale, v_j_quat);

            const auto S_ij = (i_cov_i.get_covariance_matrix() + i_cov_j.get_covariance_matrix()) / 2.0;
            Eigen::Vector3d diff = MapConst(glm::dvec3(graph.vpoint[v_i]) - glm::dvec3(graph.vpoint[v_j]));

            const double weight = std::exp(
                (-diff.transpose() * S_ij.inverse() * diff / (2.0 * double(t_factor))).value());

            auto v_i_idx = graph.from_vertex(h0).idx();
            auto v_j_idx = graph.to_vertex(h0).idx();
            // Off-diagonal elements are -1 for every connected edge
            S_triplets.emplace_back(v_i_idx, v_j_idx, -weight);
            S_triplets.emplace_back(v_j_idx, v_i_idx, -weight);


            M_diagonal_values[v_i_idx] += weight;
            M_diagonal_values[v_j_idx] += weight;
        }

        for (long i = 0; i < n_vertices; ++i) {
            S_triplets.emplace_back(i, i, M_diagonal_values[i]);
        }

        // --- Step 4: Finalize the Mass Matrix ---
        std::vector<Eigen::Triplet<float> > M_triplets;
        M_triplets.reserve(n_vertices);
        for (long i = 0; i < n_vertices; ++i) {
            if (M_diagonal_values[i] > 1e-9) {
                M_triplets.emplace_back(i, i, M_diagonal_values[i]);
            } else {
                M_triplets.emplace_back(i, i, 1e-9);
            }
        }

        matrices.build(S_triplets, M_triplets, n_vertices);
        return matrices;
    }
}

#endif //ENGINE24_SURFACEMESHLAPLACIANOPERATOR_H
