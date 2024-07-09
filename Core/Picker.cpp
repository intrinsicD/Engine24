//
// Created by alex on 03.07.24.
//

#include "Picker.h"
#include "Engine.h"
#include "imgui.h"
#include "../Camera.h"
#include "Graphics.h"
#include "Mouse.h"
#include "EventsCallbacks.h"
#include "Eigen/Geometry"

namespace Bcg {

    Vector<float, 3> screen_to_ndc(const Vector<int, 4> &viewport, float x, float y, float z) {
        float xf = ((float) x - (float) viewport[0]) / ((float) viewport[2]) * 2.0f - 1.0f;
        float yf = ((float) y - (float) viewport[1]) / ((float) viewport[3]) * 2.0f - 1.0f;
        float zf = z * 2.0f - 1.0f;
        return {xf, yf, zf};
    }

    Picker::Picker() : Plugin("Picker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
    }

    Picked &Picker::pick(double x, double y) {
        auto &picked = last_picked();
        picked.screen_space_point = {x, y};
        picked.entity.is_background = true;
        float zf;
        if (Graphics::read_depth_buffer(picked.screen_space_point[0], picked.screen_space_point[1], zf)) {
            picked.entity.is_background = false;
            Vector<int, 4> viewport = Graphics::get_viewport();
            picked.ndc_space_point = screen_to_ndc(viewport, picked.screen_space_point[0], picked.screen_space_point[1],
                                                   zf);

            auto &camera = Engine::Context().get<Camera>();

            Matrix<float, 4, 4> vp = camera.proj * camera.view;
            Matrix<float, 4, 4> inv = vp.inverse();
            Vector<float, 4> p = inv * picked.world_space_point.homogeneous();
            picked.world_space_point = p.head<3>() / p[3];
            picked.view_space_point = (camera.view * picked.world_space_point.homogeneous()).head<3>();
        }

        return picked;
    }

    Picked &Picker::last_picked() {
        return Engine::Context().get<Picked>();
    }

    static void on_mouse_button(const Events::Callback::MouseButton &event) {
        auto &mouse = Engine::Context().get<Mouse>();
        Picker::pick(mouse.cursor.xpos, mouse.cursor.ypos);
    }


    void Picker::activate() {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
        Engine::Dispatcher().sink<Events::Callback::MouseButton>().connect<&on_mouse_button>();
        Plugin::activate();
    }

    void Picker::begin_frame() {}

    void Picker::update() {}

    void Picker::end_frame() {}

    void Picker::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::MouseButton>().disconnect<&on_mouse_button>();
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void Picker::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void Picker::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = last_picked();
                Gui::Show(picked);
                ImGui::End();
            }
        }
    }

    void Picker::render() {}

    namespace Gui {
        void Show(const Picked &picked) {

            if (ImGui::CollapsingHeader("Entity")) {
                Show(picked.entity);
            }
            ImGui::Text("World Space Point: (%f, %f, %f)", picked.world_space_point.x(), picked.world_space_point.y(),
                        picked.world_space_point.z());
            ImGui::Text("View Space Point: (%f, %f, %f)", picked.view_space_point.x(), picked.view_space_point.y(),
                        picked.view_space_point.z());
            ImGui::Text("NDC Space Point: (%f, %f, %f)", picked.ndc_space_point.x(), picked.ndc_space_point.y(),
                        picked.ndc_space_point.z());
            ImGui::Text("Screen Space Point: (%f, %f)", picked.screen_space_point.x(), picked.screen_space_point.y());
        }

        void Show(const Picked::Entity &entity) {
            ImGui::Text("entity id: %u", static_cast<entt::id_type>(entity.id));
            ImGui::Text("is_background: %s", entity.is_background ? "true" : "false");
            ImGui::Text("vertex_idx: %u", entity.vertex_idx);
            ImGui::Text("edge_idx: %u", entity.edge_idx);
            ImGui::Text("face_idx: %u", entity.face_idx);
        }
    }


}