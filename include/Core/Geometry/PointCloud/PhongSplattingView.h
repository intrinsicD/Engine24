//
// Created by alex on 04.06.25.
//

#ifndef ENGINE24_PHONGSPLATTINGVIEW_H
#define ENGINE24_PHONGSPLATTINGVIEW_H

#include "VertexArrayObject.h"
#include "Program.h"
#include "Material.h"
#include "MatVec.h"

namespace Bcg {
    struct PhongSplattingView {
        Attribute position{0, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "position", ""};
        Attribute color{1, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "color", ""};
        Attribute normal{2, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "normal", ""};
        Attribute radius{3, 1, Attribute::Type::FLOAT, false, sizeof(float), "radius", ""};

        VertexArrayObject vao;
        Program program;
        unsigned int num_points;

        bool use_uniform_color = true;
        Vector<float, 3> uniform_color{1.0, 1.0, 1.0};
        bool use_uniform_radius = true;
        float uniform_radius = 0.001;

        unsigned int material_id = -1;

        Vector<float, 3> uLightColor = {1.0f, 1.0f, 1.0f};
        Vector<float, 3> uAmbientColor = {0.1f, 0.1f, 0.1f};
        Vector<float, 3> uSpecularColor = {1.0f, 1.0f, 1.0f};
        float uShininess = 32.0f;

        float min_color = 0;
        float max_color = 1;
        bool hide = false;
    };
}
#endif //ENGINE24_PHONGSPLATTINGVIEW_H
