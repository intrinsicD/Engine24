#pragma once

#include "AABBUtils.h"
#include "GraphInterface.h"

namespace Bcg {
    inline void AABBsToGraph(const std::vector<AABB<float> > &aabbs, GraphInterface &out_graph) {
        if (aabbs.empty()) return;
        size_t offset = 0;
        auto aabb_edges = AABBUtils::GetEdges(aabbs[0]);
        for (const auto &aabb: aabbs) {
            auto aabb_vertices = AABBUtils::GetVertices(aabb);
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

            // --- CRITICAL STEP: Construct a proper quaternion ---
            // Assuming the vec4 stores (x, y, z, w). The glm::quat constructor is (w, x, y, z).
            //glm::qua<T> rotation = glm::normalize(glm::qua<T>(rot_vec.w, rot_vec.x, rot_vec.y, rot_vec.z));
            glm::qua<T> rotation = glm::normalize(glm::qua<T>(rot_vec.x, rot_vec.y, rot_vec.z, rot_vec.w));
            // If you discover your data is actually (w, x, y, z), the constructor is simpler:
            // glm::qua<T> rotation = glm::normalize(glm::qua<T>(rot_vec.w, rot_vec.x, rot_vec.y, rot_vec.z));
            // Or even more directly from the vec4 components:
            // glm::qua<T> rotation = glm::normalize(glm::make_quat(&rot_vec[0])); // Be cautious with this one.

            // 1. Add vertices for the current box to the graph.
            for (const auto &local_v_f: CANONICAL_BOX_VERTICES) {
                glm::vec<3, T> v = static_cast<glm::vec<3, T>>(local_v_f);

                // Apply transformation pipeline: Scale -> Rotate -> Translate
                v = v * scale; // Component-wise scaling
                v = rotation * v; // Correct, unambiguous quaternion rotation
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
}
