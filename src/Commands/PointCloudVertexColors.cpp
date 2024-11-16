//
// Created by alex on 11/10/24.
//

#include "PointCloudVertexColors.h"
#include "GetPrimitives.h"
#include "Engine.h"
#include "SphereView.h"
#include "OpenGLState.h"
#include "PropertyEigenMap.h"
#include "Pool.h"


namespace Bcg::Commands {
    void SetPointCloudVertexColors3D::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;



        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        if(property_name.empty()) {
            property_name = std::string("v_colors");
        }

        auto v_color = vertices->get<Vector<float, 3>>(property_name);
        auto b_color = openGlState.get_buffer(property_name);

        if (v_color) {
            if (!b_color) {
                b_color = ArrayBuffer();
                b_color.create();
                openGlState.register_buffer(property_name, b_color);
            }

            view.vao.bind();
            b_color.bind();
            b_color.buffer_data(v_color.data(),
                                num_vertices * 3 * sizeof(float),
                                Buffer::STATIC_DRAW);
            view.min_color = Map(v_color.vector()).minCoeff();
            view.max_color = Map(v_color.vector()).maxCoeff();
            if(view.min_color == view.max_color){
                view.min_color = 0;
            }

            view.color.set(nullptr);
            view.color.enable();
            view.use_uniform_color = false;
            view.vao.unbind();
        } else {
            view.use_uniform_color = true;
        }
        view.color.bound_buffer_name = property_name.c_str();


        if (v_color) {
            b_color.unbind();
        }
    }

    void SetPointCloudVertexColorsScalarfield::execute() const {
        //TODO
    }

    void SetPointCloudVertexColorsSelection3D::execute() const {

    }

    void SetPointCloudVertexColorsSelectionScalarfield::execute() const {

    }

}