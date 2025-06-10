//
// Created by alex on 06.06.25.
//

#ifndef ENGINE24_GRAPHVIEW_H
#define ENGINE24_GRAPHVIEW_H
#include "SphereView.h"
#include "Program.h"
#include "MatVec.h"
#include "Buffer.h"

namespace Bcg{
    struct GraphView {
        Attribute position{0, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "position", ""};
        Attribute color{1, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "edge_color", ""};
        Attribute scalarfield{2, 3, Attribute::Type::FLOAT, false, sizeof(float), "edge_scalarfield", ""};

        bool use_scalarfield = false;

        VertexArrayObject vao;
        Program program;
        unsigned int num_indices;

        Vector<float, 3> uniform_color{1.0, 1.0, 1.0};
        bool use_uniform_color = true;
        float min_color = 0;
        float max_color = 1;
        bool hide = false;
    };
}
#endif //ENGINE24_GRAPHVIEW_H
