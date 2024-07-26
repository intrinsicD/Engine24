//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_VIEWS_H
#define ENGINE24_VIEWS_H

#include "VertexArrayObject.h"
#include "Buffer.h"
#include "Program.h"
#include "MatVec.h"
#include "DrawCall.h"
#include "entt/fwd.hpp"

namespace Bcg {
    struct EntityView{
        VertexArrayObject vao;
        Program program;
        unsigned int num_elements;
        //and then a mapping which attributes are set to which buffer
    };

    struct PointView {
        VertexArrayObject vao;
        ArrayBuffer vbo;
        BufferLayout layout;
        Program program;
        unsigned int num_indices;
    };

    struct PointCloudView : public PointView {

    };

    struct LineView {
        VertexArrayObject vao;
        ElementArrayBuffer ebo;
        ArrayBuffer vbo;
        BufferLayout layout;
        Program program;
        unsigned int num_indices;
    };

    struct GraphView : public LineView {

    };

    struct TriangleView {
        VertexArrayObject vao;
        Program program;
        unsigned int num_indices;

        virtual void draw();
    };

    struct PickingView {
        VertexArrayObject vao;
        Program program;
        unsigned int num_indices;

        Vector<float, 4> picking_color;

        Vector<float, 4> encode(entt::entity entity_id);

        entt::entity encode(const Vector<float, 4> &picking_color);

        void draw();
    };

    struct MeshView : public TriangleView {

    };
}

#endif //ENGINE24_VIEWS_H
