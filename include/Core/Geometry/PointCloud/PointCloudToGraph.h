#pragma once

#include "PointCloudInterface.h"
#include "Graph.h"

namespace Bcg {
    void PointCloudToKNNGraph(PointCloudInterface &pci, int k, GraphInterface &out_graph);

    void PointCloudToRadiusGraph(PointCloudInterface &pci, float radius, GraphInterface &out_graph);
}
