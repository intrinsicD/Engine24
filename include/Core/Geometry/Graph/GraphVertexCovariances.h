#pragma once
#include "Graph.h"
#include "MatVec.h"
#include "../GeometryCommon.h"
#include "../../Types.h"
#include "eigen3/Eigen/Dense"

namespace Bcg {
    template<typename T>
    VertexProperty<Matrix<T, 3> > GetVertexCovariances(Graph &graph, VertexProperty<PointType> means) {
        VertexProperty<Matrix<T, 3> > covariances = graph.vprops_.get_or_add<Matrix<T, 3> >(
            "v_covariance", Matrix<T, 3>::Zero());
        VertexProperty<int> counts = graph.vprops_.get_or_add<int>("v_counts", 0);
        for (const auto v: graph.vertices()) {
            covariances[v] = Matrix<T, 3>::Zero();
            for (const auto &vv: graph.get_vertices(v)) {
                Vector<T, 3> diff = graph.positions[vv] - means[v];
                covariances[v] += diff * diff.transpose();
                ++counts[v];
            }
            if (counts[v] > 0) {
                covariances[v] /= T(counts[v]);
            }
        }
        graph.remove_vertex_property(covariances);
        return covariances;
    }
}
