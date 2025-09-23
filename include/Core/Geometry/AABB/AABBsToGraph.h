#pragma once

#include "AABBUtils.h"
#include "GraphInterface.h"

namespace Bcg {
    void AABBsToGraph(const std::vector<AABB<float> > &aabbs, GraphInterface &out_graph) {
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
}
