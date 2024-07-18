//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_VIEWS_H
#define ENGINE24_VIEWS_H

#include "VertexArrayObject.h"
#include "Buffer.h"
#include "Shader.h"

namespace Bcg{
    struct PointView{
        VertexArrayObject vao;
        ArrayBuffer vbo;
        BufferLayout layout;
        Program program;
        unsigned int num_indices;
    };

    struct LineView {
        VertexArrayObject vao;
        ElementArrayBuffer ebo;
        ArrayBuffer vbo;
        BufferLayout layout;
        Program program;
        unsigned int num_indices;
    };

    struct TriangleView {
        VertexArrayObject vao;
        ElementArrayBuffer ebo;
        ArrayBuffer vbo;
        BufferLayout layout;
        Program program;
        unsigned int num_indices;

        void draw();
    };

    struct MeshView : public TriangleView{

    };
}

#endif //ENGINE24_VIEWS_H
