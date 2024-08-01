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
    struct EntityView {
        VertexArrayObject vao;
        Program program;
        unsigned int num_elements;
        //and then a mapping which attributes are set to which buffer
    };

    struct PointView {
        VertexArrayObject vao;
        Program program;
        unsigned int offset = 0;
        unsigned int num_indices;
        bool hide = false;

        virtual void draw();
    };

    struct PointCloudView : public PointView {
        Vector<float, 3> base_color = {0.8, 0.8, 0.8};
    };

    struct LineView {
        VertexArrayObject vao;
        Program program;
        unsigned int num_indices;
        bool hide = false;
    };

    struct GraphView : public LineView {

    };

    struct TriangleView {
        VertexArrayObject vao;
        Program program;
        unsigned int num_indices;
        bool hide = false;

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
        Vector<float, 3> base_color = {0.3, 0.6, 0.2};
    };

    struct VectorfieldView : public PointView {

    };

    struct VectorfieldViews : public std::unordered_map<std::string, VectorfieldView> {
        using std::unordered_map<std::string, VectorfieldView>::unordered_map;
        bool hide = false;
    };
}

#endif //ENGINE24_VIEWS_H
