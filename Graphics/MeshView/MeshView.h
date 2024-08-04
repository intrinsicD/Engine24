//
// Created by alex on 04.08.24.
//

#ifndef ENGINE24_MESHVIEW_H
#define ENGINE24_MESHVIEW_H

#include "VertexArrayObject.h"
#include "Program.h"
#include "MatVec.h"

namespace Bcg{
    struct MeshView {
        Attribute position{0, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "position", ""};
        Attribute normal{1, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "normal", ""};
        Attribute color{2, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "color", ""};

        VertexArrayObject vao;
        Program program;
        unsigned int num_indices;

        Vector<float, 3> base_color{0.8, 0.8, 0.8};
        float min_color = 0;
        float max_color = 1;
        bool hide = false;

        void draw();
    };
}

#endif //ENGINE24_MESHVIEW_H
