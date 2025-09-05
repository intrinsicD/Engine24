#pragma once

#include "Graph.h"
#include "GeometryCommon.h"
#include "Types.h"

namespace Bcg {
    Graph PointCloudToKNNGraph(const VertexProperty<PointType> &points, int k);

    Graph PointCloudToRadiusGraph(const VertexProperty<PointType> &points, float radius);
}
