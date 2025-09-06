#include "PointCloudToGraph.h"
#include "KDTreeCpu.h"

namespace Bcg {
    Graph ToGraph(const VertexProperty<PointType> &points, const std::vector<QueryResult> &results) {
        Graph graph;
        for (size_t i = 0; i < points.vector().size(); ++i) {
            graph.interface.add_vertex(points.vector()[i]);
        }
        for (size_t v_i = 0; v_i < results.size(); ++v_i) {
            const auto &res = results[v_i];
            for (size_t j = 0; j < res.size(); ++j) {
                const auto v_j = res.indices[j];
                if (v_j == v_i) continue; // skip self-loop
                graph.interface.add_edge(Vertex(v_i), Vertex(v_j));
            }
        }
        return graph;
    }

    Graph PointCloudToKNNGraph(const VertexProperty<PointType> &points, int k) {
        KDTreeCpu kdtree;
        kdtree.build(points.vector());

        auto results = kdtree.knn_query_batch(points.vector(), k);

        return ToGraph(points, results);
    }

    Graph PointCloudToRadiusGraph(const VertexProperty<PointType> &points, float radius) {
        KDTreeCpu kdtree;
        kdtree.build(points.vector());

        auto results = kdtree.radius_query_batch(points.vector(), radius);

        return ToGraph(points, results);
    }
}
