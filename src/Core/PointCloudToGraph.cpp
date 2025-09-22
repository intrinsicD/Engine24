#include "PointCloudToGraph.h"

#include "AABBUtils.h"
#include "KDTreeCpu.h"
#include "Octree.h"
#include "Timer.h"

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

    void ToGraph(GraphInterface &graph, const std::vector<std::vector<size_t> > &results) {
        for (size_t v_i = 0; v_i < results.size(); ++v_i) {
            const auto &res = results[v_i];
            for (size_t j = 0; j < res.size(); ++j) {
                const auto v_j = res[j];
                if (v_j == v_i) continue; // skip self-loop
                graph.add_edge(Vertex(v_i), Vertex(v_j));
            }
        }
    }

    void PointCloudToKNNGraph(PointCloudInterface &pci, int k, GraphInterface &out_graph) {
        Timer timer;
        timer.start();
        KDTreeCpu kdtree;
        kdtree.build(pci.vpoint.vector());

        Log::Info("KDTREE BUILD CPU: " + std::to_string(timer.stop().delta) + " seconds");
        timer.start();

        auto results = kdtree.knn_query_batch(pci.vpoint.vector(), k);

        Log::Info("KDTREE KNN CPU: " + std::to_string(timer.stop().delta) + " seconds");

        auto aabbs = pci.vertex_property<AABB<float> >("v:aabb");
        aabbs.vector() = AABBUtils::ConvertToAABBs(pci.vpoint.vector());

        timer.start();
        Octree octree;
        octree.build(aabbs, {Octree::SplitPoint::Median, true, 0.0f}, 32, 10);

        Log::Info("OCTREE BUILD CPU: " + std::to_string(timer.stop().delta) + " seconds");
        timer.start();

        std::vector<std::vector<size_t> > res;
        res.resize(pci.vpoint.vector().size());
        for (const auto v: pci.vertices) {
            octree.query_knn(pci.vpoint[v], k, res[v.idx()]);
        }
        Log::Info("OCTREE KNN CPU: " + std::to_string(timer.stop().delta) + " seconds");

        out_graph.vertices.remove_vertex_property(out_graph.vconnectivity);
        out_graph.halfedges.remove_halfedge_property(out_graph.hconnectivity);
        out_graph.vconnectivity = out_graph.vertices.vertex_property<Halfedge>("v:connectivity");
        out_graph.hconnectivity = out_graph.halfedges.halfedge_property<HalfedgeConnectivity>("h:connectivity");
        ToGraph(out_graph, res);
    }

    void PointCloudToRadiusGraph(PointCloudInterface &pci, float radius, GraphInterface &out_graph) {
        KDTreeCpu kdtree;
        kdtree.build(pci.vpoint.vector());

        auto results = kdtree.radius_query_batch(pci.vpoint.vector(), radius);

        out_graph.vertices.remove_vertex_property(out_graph.vconnectivity);
        out_graph.halfedges.remove_halfedge_property(out_graph.hconnectivity);
        out_graph.vconnectivity = out_graph.vertices.vertex_property<Halfedge>("v:connectivity");
        out_graph.hconnectivity = out_graph.halfedges.halfedge_property<HalfedgeConnectivity>("h:connectivity");
        ToGraph(out_graph, results);
    }
}
