//
// Created by alex on 22.08.24.
//

#ifndef ENGINE24_GEOMETRYDATA_H
#define ENGINE24_GEOMETRYDATA_H

#include "Properties.h"

namespace Bcg {
    struct Vertices : public PropertyContainer {
    };

    struct HalfEdges : public PropertyContainer {
    };

    struct Edges : public PropertyContainer {
    };

    struct Faces : public PropertyContainer {
    };

    struct PointCloudData{
        Vertices vertices;
    };

    struct GraphData{
        Vertices vertices;
        HalfEdges halfedges;
        Edges edges;
    };

    struct MeshData{
        Vertices vertices;
        HalfEdges halfedges;
        Edges edges;
        Faces faces;
    };
}

#endif //ENGINE24_GEOMETRYDATA_H
