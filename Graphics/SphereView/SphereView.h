//
// Created by alex on 02.08.24.
//

#ifndef ENGINE24_SPHEREVIEW_H
#define ENGINE24_SPHEREVIEW_H

#include "VertexArrayObject.h"
#include "Program.h"
#include "MatVec.h"

namespace Bcg {
    struct SphereView {
        Attribute position{0, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "position", ""};
        Attribute color{1, 3, Attribute::Type::FLOAT, false, 3 * sizeof(float), "color", ""};
        Attribute radius{2, 1, Attribute::Type::FLOAT, false, sizeof(float), "radius", ""};

        VertexArrayObject vao;
        Program program;
        unsigned int num_spheres;

        Vector<float, 3> base_color{0.8, 0.8, 0.8};
        float default_radius = 10.0;
        float min_color = 0;
        float max_color = 1;

        void draw();
    };
}

#endif //ENGINE24_SPHEREVIEW_H
