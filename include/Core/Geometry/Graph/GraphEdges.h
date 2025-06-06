//
// Created by alex on 06.06.25.
//

#ifndef ENGINE24_GRAPHEDGES_H
#define ENGINE24_GRAPHEDGES_H

#include "GraphInterface.h"

namespace Bcg{
    inline Property<Vector<unsigned int, 2>> GraphGetEdges(GraphInterface  &graph){
            EdgeProperty<Vector<unsigned int, 2>> edges = graph.edge_property<Vector<unsigned int, 2>>("e:edges");
            for (auto e: graph.edges) {
                if(graph.edges.is_valid(e)){
                    auto v0 = graph.get_vertex(e, 0);
                    auto v1 = graph.get_vertex(e, 1);
                    if(graph.vertices.is_valid(v0) && graph.vertices.is_valid(v1)){
                        edges[e][0] = v0.idx();
                        edges[e][1] = v1.idx();
                    } else {
                        // Handle invalid vertices if necessary
                    }
                }
            }
            return edges;
    }
}

#endif //ENGINE24_GRAPHEDGES_H
