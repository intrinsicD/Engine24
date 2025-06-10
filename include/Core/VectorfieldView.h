//
// Created by alex on 05.08.24.
//

#ifndef ENGINE24_VECTORFIELDVIEW_H
#define ENGINE24_VECTORFIELDVIEW_H


#include "VertexArrayObject.h"
#include "Program.h"
#include "MatVec.h"

namespace Bcg {
    struct VectorfieldView {
        Attribute position{0, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "position", ""};
        Attribute color{1, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "color", ""};
        Attribute vector{2, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "vector", ""};
        Attribute length{3, 1, Attribute::Type::FLOAT, false, sizeof(float), "length", ""};

        VertexArrayObject vao;
        Program program;
        unsigned int num_vectors;

        Vector<float, 3> uniform_color{1.0, 1.0, 1.0};
        bool use_uniform_color = true;
        float uniform_length = 1;
        bool use_uniform_length = true;
        float min_color = 0;
        float max_color = 1;
        bool hide = false;

        std::string vectorfield_name;
    };

    struct VectorfieldViews {
        std::unordered_map<std::string, VectorfieldView> vectorfields;
        bool hide = false;
    };
}

#endif //ENGINE24_VECTORFIELDVIEW_H
