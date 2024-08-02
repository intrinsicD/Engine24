//
// Created by alex on 02.08.24.
//

#include "PluginSphereView.h"
#include "Engine.h"
#include "imgui.h"
#include "SphereViewGui.h"
#include "Picker.h"
#include "Camera.h"
#include "Graphics.h"
#include "Transform.h"
#include "EventsCallbacks.h"
#include "Keyboard.h"
#include "glad/gl.h"

namespace Bcg {
    static float global_point_size = 1.0f;

    static void on_mouse_scroll(const Events::Callback::MouseScroll &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (!keyboard.strg()) return;
        auto &picked = Engine::Context().get<Picked>();
        auto entity_id = picked.entity.id;
        if (!picked.entity.is_background && Engine::has<SphereView>(entity_id)) {
            auto &view = Engine::State().get<SphereView>(entity_id);
            view.default_radius = std::max<float>(1.0f, view.default_radius + event.yoffset);
        } else {
            global_point_size = std::max<float>(1.0f, global_point_size + event.yoffset);
            glPointSize(global_point_size);
        }
    }

    void PluginSphereView::activate() {
        Plugin::activate();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&on_mouse_scroll>();
    }

    void PluginSphereView::begin_frame() {}

    void PluginSphereView::update() {}

    void PluginSphereView::end_frame() {}

    void PluginSphereView::deactivate() {
        Plugin::deactivate();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().disconnect<&on_mouse_scroll>();
    }

    static bool show_gui = false;

    void PluginSphereView::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("SphereView", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginSphereView::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("SphereView", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowSphereView(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginSphereView::render() {
        auto rendergroup = Engine::State().view<SphereView>();
        auto &camera = Engine::Context().get<Camera>();
        auto vp = Graphics::get_viewport();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<SphereView>(entity_id);

            view.vao.bind();
            view.program.use();
            view.program.set_uniform3fv("light_position", camera.v_params.eye.data());
            view.program.set_uniform1ui("width", vp[2]);
            view.program.set_uniform1ui("height", vp[3]);
            view.program.set_uniform1f("pointSize", view.default_radius);
            view.program.set_uniform1f("min_color", view.min_color);
            view.program.set_uniform1f("max_color", view.max_color);

            if (Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                view.program.set_uniform4fm("model", transform.data(), false);
            } else {
                view.program.set_uniform4fm("model", Transform().data(), false);
            }

            view.draw();
        }
    }
}