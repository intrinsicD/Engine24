#pragma once

#include "AABBUtils.h"
#include "GraphInterface.h"

namespace Bcg {
    inline void AABBsToGraph(const std::vector<AABB<float> > &aabbs, GraphInterface &out_graph) {
        if (aabbs.empty()) return;
        size_t offset = 0;
        auto aabb_edges = GetEdges(aabbs[0]);
        for (const auto &aabb: aabbs) {
            auto aabb_vertices = GetVertices(aabb);
            for (const auto &v: aabb_vertices) {
                out_graph.add_vertex(v);
            }
            for (const auto &e: aabb_edges) {
                out_graph.add_edge(Vertex(e[0] + offset), Vertex(e[1] + offset));
            }
            offset += aabb_vertices.size();
        }
    }

    template<typename T>
    Vector<T, 3> rotate_vector_by_quaternion(const Vector<T, 3> &v, const Vector<T, 4> &q) {
        Vector<T, 3> q_vec(q.x, q.y, q.z);
        Vector<T, 3> t = static_cast<T>(2.0) * glm::cross(q_vec, v);
        return v + q.w * t + glm::cross(q_vec, t);
    }

    const std::vector<glm::vec3> CANONICAL_BOX_VERTICES = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}
    };

    const std::vector<std::array<unsigned int, 2> > CANONICAL_BOX_EDGES = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    // Existing API: scales interpreted as standard deviations (sigma);
    // half-extent equals 0.5 * scales (historical behavior). Now corrected quaternion order.
    template<typename T>
    void OrientedBoxesToGraph(const std::vector<glm::vec<3, T> > &means,
                              const std::vector<glm::vec<3, T> > &scales,
                              const std::vector<glm::vec<4, T> > &rotations_vec4, // Input remains vec4
                              GraphInterface &out_graph) {
        if (means.empty()) return;

        size_t vertex_offset = out_graph.vertices.n_vertices(); // Use the actual graph's vertex count

        for (size_t i = 0; i < means.size(); ++i) {
            const auto &mean = means[i];
            const auto &scale = scales[i];
            const auto &rot_vec = rotations_vec4[i];

            // Construct proper quaternion. Throughout the codebase, rotation is stored as (w, x, y, z).
            Matrix<float, 3, 3> rotation = glm::mat3_cast(glm::quat(rot_vec.x, rot_vec.y, rot_vec.z, rot_vec.w));
            // 1. Add vertices for the current box to the graph.
            for (const auto &local_v_f: CANONICAL_BOX_VERTICES) {
                glm::vec<3, T> v = static_cast<glm::vec<3, T> >(local_v_f);

                // Apply transformation pipeline: Scale -> Rotate -> Translate
                v = v * scale; // Component-wise scaling
                v = rotation * v; // Quaternion rotation
                v = v + mean; // Translation

                out_graph.add_vertex(v);
            }

            // 2. Add edges for the current box, adjusted by the offset.
            for (const auto &edge_indices: CANONICAL_BOX_EDGES) {
                out_graph.add_edge(Vertex(edge_indices[0] + vertex_offset),
                                   Vertex(edge_indices[1] + vertex_offset));
            }

            // 3. Update the offset for the next box.
            vertex_offset += CANONICAL_BOX_VERTICES.size();
        }
    }

    // New API: explicitly control the half-extent in units of sigma via sigma_k (e.g., k=3 for 3-sigma).
    // If sigma_k = 3, the full edge length will be 6*sigma because the canonical box is [-0.5, 0.5].
    template<typename T>
    void OrientedBoxesToGraph(const std::vector<glm::vec<3, T> > &means,
                              const std::vector<glm::vec<3, T> > &scales,
                              const std::vector<glm::vec<4, T> > &rotations_vec4,
                              GraphInterface &out_graph,
                              T sigma_k) {
        if (means.empty()) return;

        size_t vertex_offset = out_graph.vertices.n_vertices();
        const T mul = static_cast<T>(2) * sigma_k; // so half-extent = sigma_k * sigma

        for (size_t i = 0; i < means.size(); ++i) {
            const auto &mean = means[i];
            const auto &scale = glm::exp(scales[i]) * mul; // scale -> 2*k*sigma
            const auto &rot_vec = rotations_vec4[i];

            glm::qua<T> rotation = glm::normalize(glm::qua<T>(rot_vec.x, rot_vec.y, rot_vec.z, rot_vec.w));

            for (const auto &local_v_f: CANONICAL_BOX_VERTICES) {
                glm::vec<3, T> v = static_cast<glm::vec<3, T> >(local_v_f);
                v = v * scale;
                v = rotation * v;
                v = v + mean;
                out_graph.add_vertex(v);
            }

            for (const auto &edge_indices: CANONICAL_BOX_EDGES) {
                out_graph.add_edge(Vertex(edge_indices[0] + vertex_offset),
                                   Vertex(edge_indices[1] + vertex_offset));
            }

            vertex_offset += CANONICAL_BOX_VERTICES.size();
        }
    }
}
