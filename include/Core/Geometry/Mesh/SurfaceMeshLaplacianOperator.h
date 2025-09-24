//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_SURFACEMESHLAPLACIANOPERATOR_H
#define ENGINE24_SURFACEMESHLAPLACIANOPERATOR_H

#include "LaplacianOperator.h"
#include "SurfaceMesh.h" // Assuming this provides vertex, edge, face iterators and connectivity
#include "Eigen/Geometry"
#include <vector>

namespace Bcg {
    /**
     * @brief Computes the geometrically accurate Cotangent Laplacian and associated matrices.
     * This is the standard, preferred Laplacian for most geometry processing tasks on triangle meshes.
     * S_ij = -0.5 * (cot(alpha_ij) + cot(beta_ij)) for an edge (i,j)
     * M_ii = Voronoi area associated with vertex i (approximated by 1/3 of adjacent triangle areas).
     * @param mesh A triangle mesh with valid vertex positions.
     * @return A LaplacianMatrices struct containing the Stiffness (S) and Mass (M) matrices.
     */
    inline LaplacianMatrices ComputeCotanLaplacianOperator(SurfaceMesh &mesh) {
        LaplacianMatrices matrices;
        const long n_vertices = mesh.data.vertices.n_vertices();
        if (n_vertices == 0) return matrices;

        // Use a list of triplets to build the sparse matrices efficiently.
        std::vector<Eigen::Triplet<float> > S_triplets;

        // Use a vector to accumulate the diagonal entries for both S and M.
        Eigen::VectorXd M_diagonal_areas = Eigen::VectorXd::Zero(n_vertices);

        // Iterate over all faces to compute cotangent weights and barycentric areas.
        // This is more robust than iterating over edges for handling boundary conditions.
        for (const auto face: mesh.data.faces) {
            // Get the 3 vertices of the triangle face
            auto h = mesh.interface.get_halfedge(face);
            auto v0_h = mesh.interface.to_vertex(h);
            auto v1_h = mesh.interface.to_vertex(mesh.interface.get_next(h));
            auto v2_h = mesh.interface.to_vertex(mesh.interface.get_prev(h));

            // Get their 3D positions
            const PointType &p0 = mesh.interface.vpoint[v0_h];
            const PointType &p1 = mesh.interface.vpoint[v1_h];
            const PointType &p2 = mesh.interface.vpoint[v2_h];

            // Calculate edge vectors
            glm::dvec3 e0 = glm::dvec3(p2) - glm::dvec3(p1);
            glm::dvec3 e1 = glm::dvec3(p0) - glm::dvec3(p2);
            glm::dvec3 e2 = glm::dvec3(p1) - glm::dvec3(p0);

            // Calculate face area (using cross product)
            // The area of the triangle is 0.5 * |e2 x (-e1)|
            double twice_area = glm::length(glm::cross(e2, -e1));
            if (twice_area < 1e-9) continue; // Skip degenerate triangles
            double area = twice_area * 0.5;

            // Add 1/3 of the triangle's area to each of its vertices.
            // This is the standard barycentric area for the mass matrix.
            M_diagonal_areas[v0_h.idx()] += area / 3.0;
            M_diagonal_areas[v1_h.idx()] += area / 3.0;
            M_diagonal_areas[v2_h.idx()] += area / 3.0;

            // Calculate cotangents for the Stiffness matrix (S)
            // cot(angle) = (a.dot(b)) / |a x b| = (a.dot(b)) / (2 * Area)
            double cot0 = 0.5 * glm::dot(-e1, e2) / twice_area;
            double cot1 = 0.5 * glm::dot(-e2, e0) / twice_area;
            double cot2 = 0.5 * glm::dot(-e0, e1) / twice_area;

            // Add contributions to the stiffness matrix triplets.
            // The weight on edge (i,j) is -0.5 * (cot(alpha) + cot(beta))
            // We iterate face-by-face, so we add the contribution from one side (e.g., cot(alpha)) now.
            // The contribution from the other side (cot(beta)) will be added when we visit the adjacent face.
            S_triplets.emplace_back(v1_h.idx(), v2_h.idx(), -cot0);
            S_triplets.emplace_back(v2_h.idx(), v1_h.idx(), -cot0);
            S_triplets.emplace_back(v0_h.idx(), v2_h.idx(), -cot1);
            S_triplets.emplace_back(v2_h.idx(), v0_h.idx(), -cot1);
            S_triplets.emplace_back(v0_h.idx(), v1_h.idx(), -cot2);
            S_triplets.emplace_back(v1_h.idx(), v0_h.idx(), -cot2);

            // Accumulate diagonal entries for S. S_ii = sum of weights of incident edges.
            S_triplets.emplace_back(v0_h.idx(), v0_h.idx(), cot1 + cot2);
            S_triplets.emplace_back(v1_h.idx(), v1_h.idx(), cot0 + cot2);
            S_triplets.emplace_back(v2_h.idx(), v2_h.idx(), cot0 + cot1);
        }

        // Finalize the Mass Matrix (M)
        std::vector<Eigen::Triplet<float> > M_triplets;
        M_triplets.reserve(n_vertices);
        for (long i = 0; i < n_vertices; ++i) {
            // Ensure mass is non-zero to avoid division by zero later
            if (M_diagonal_areas[i] > 1e-9) {
                M_triplets.emplace_back(i, i, M_diagonal_areas[i]);
            } else {
                M_triplets.emplace_back(i, i, 1e-9);
            }
        }

        matrices.build(S_triplets, M_triplets, n_vertices);
        return matrices;
    }

    /**
     * @brief Computes the simple Graph Laplacian (Combinatorial Laplacian).
     * This operator only considers connectivity, not geometry.
     * S_ij = -1 if (i,j) is an edge, 0 otherwise.
     * S_ii = degree of vertex i.
     * The Mass matrix M is the identity matrix.
     * @param mesh A surface mesh.
     * @return A LaplacianMatrices struct containing the Stiffness (S) and Mass (M) matrices.
     */
    inline LaplacianMatrices ComputeGraphLaplacianOperator(SurfaceMesh &mesh) {
        LaplacianMatrices matrices;
        const long n_vertices = mesh.data.vertices.n_vertices();
        if (n_vertices == 0) return matrices;

        std::vector<Eigen::Triplet<float> > S_triplets;
        // Reserve space: 2 triplets for each edge (i,j) and (j,i), plus n for the diagonal.
        S_triplets.reserve(mesh.data.edges.n_edges() * 2 + n_vertices);

        Eigen::VectorXf diagonal_degrees = Eigen::VectorXf::Zero(n_vertices);

        // Iterate over all edges to set off-diagonal entries
        for (const auto edge: mesh.data.edges) {
            auto h0 = mesh.interface.get_halfedge(edge, 0);
            auto v_i = mesh.interface.from_vertex(h0);
            auto v_j = mesh.interface.to_vertex(h0);

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

        std::vector<Eigen::Triplet<float> > M_triplets;
        M_triplets.reserve(n_vertices);
        for (int i = 0; i < n_vertices; ++i) M_triplets.emplace_back(i, i, 1.0f);

        matrices.build(S_triplets, M_triplets, n_vertices);

        return matrices;
    }
}

#endif //ENGINE24_SURFACEMESHLAPLACIANOPERATOR_H
