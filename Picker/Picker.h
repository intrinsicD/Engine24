//
// Created by alex on 03.07.24.
//

#ifndef ENGINE24_PICKER_H
#define ENGINE24_PICKER_H

#include "entt/fwd.hpp"
#include "MatVec.h"
#include "Plugin.h"
#include "GuiUtils.h"

namespace Bcg {
    struct Picked {
        struct Entity {
            entt::entity id;
            bool is_background = true;

            operator bool() { return !is_background; }

            unsigned int vertex_idx = -1;
            unsigned int edge_idx = -1;
            unsigned int face_idx = -1;
            Vector<float, 3> model_space_point;
        } entity;

        Vector<float, 3> world_space_point;
        Vector<float, 3> view_space_point;
        Vector<float, 3> ndc_space_point;
        Vector<float, 2> screen_space_point;
    };

    Vector<float, 3>
    world_to_model(const Matrix<float, 4, 4> &model_inverse_matrix, const Vector<float, 3> &world_space_point);

    Vector<float, 3>
    model_to_world(const Matrix<float, 4, 4> &model_matrix, const Vector<float, 3> &model_space_point);

    Vector<float, 3>
    view_to_world(const Matrix<float, 4, 4> &view_inverse_matrix, const Vector<float, 3> &view_space_point);

    Vector<float, 3>
    world_to_wiew(const Matrix<float, 4, 4> &view_matrix, const Vector<float, 3> &world_space_point);

    Vector<float, 3> screen_to_ndc(const Vector<int, 4> &viewport, float x, float y, float z /*depth_buffer_value*/);


    class Picker : public Plugin {
    public:
        Picker();

        ~Picker() override = default;

        static Picked &pick(double x, double y);

        static Picked &last_picked();

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };

    namespace Gui {
        void Show(const Picked &picked);

        void Show(const Picked::Entity &entity);
    };
}

#endif //ENGINE24_PICKER_H
