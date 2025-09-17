#include "PointCloudToGraph.h"
#include "KDTreeCpu.h"

namespace Bcg {
    void ToGraph(GraphInterface &graph, const std::vector<QueryResult> &results) {
        for (size_t v_i = 0; v_i < results.size(); ++v_i) {
            const auto &res = results[v_i];
            for (size_t j = 0; j < res.size(); ++j) {
                const auto v_j = res.indices[j];
                if (v_j == v_i) continue; // skip self-loop
                graph.add_edge(Vertex(v_i), Vertex(v_j));
            }
        }
    }

    void PointCloudToKNNGraph(PointCloudInterface &pci, int k, GraphInterface &out_graph) {
        KDTreeCpu kdtree;
        kdtree.build(pci.vpoint.vector());

        auto results = kdtree.knn_query_batch(pci.vpoint.vector(), k);

        ToGraph(out_graph, results);
    }

    void PointCloudToRadiusGraph(PointCloudInterface &pci, float radius, GraphInterface &out_graph) {
        KDTreeCpu kdtree;
        kdtree.build(pci.vpoint.vector());

        auto results = kdtree.radius_query_batch(pci.vpoint.vector(), radius);

        ToGraph(out_graph, results);
    }
}
