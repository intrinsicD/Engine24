//
// Created by alex on 05.08.24.
//

#include "PluginVectorfieldViews.h"
#include "Engine.h"
#include "imgui.h"
#include "VectorfieldViewGui.h"
#include "Picker.h"
#include "Camera.h"
#include "Graphics.h"
#include "Transform.h"

namespace Bcg {
    void PluginVectorfieldViews::activate() {
        Plugin::activate();
    }

    void PluginVectorfieldViews::begin_frame() {}

    void PluginVectorfieldViews::update() {}

    void PluginVectorfieldViews::end_frame() {}

    void PluginVectorfieldViews::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginVectorfieldViews::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("VectorfieldViews", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginVectorfieldViews::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("VectorfieldViews", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowVectorfieldViews(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginVectorfieldViews::render() {
        auto rendergroup = Engine::State().view<VectorfieldViews>();
        auto &camera = Engine::Context().get<Camera>();
        auto vp = Graphics::get_viewport();
        for (auto entity_id: rendergroup) {
            auto &views = Engine::State().get<VectorfieldViews>(entity_id);
            if (views.hide) continue;
            for (auto &item: views.vectorfields) {
                auto &view = item.second;

                if (view.hide) continue;

                view.vao.bind();
                view.program.use();
                view.program.set_uniform1i("use_uniform_length", view.use_uniform_length);
                view.program.set_uniform1f("uniform_length", view.uniform_length);
                view.program.set_uniform1f("min_color", view.min_color);
                view.program.set_uniform1f("max_color", view.max_color);
                view.program.set_uniform1i("use_uniform_color", view.use_uniform_color);
                view.program.set_uniform3fv("uniform_color", view.uniform_color.data());

                if (Engine::has<Transform>(entity_id)) {
                    auto &transform = Engine::State().get<Transform>(entity_id);
                    view.program.set_uniform4fm("model", transform.data(), false);
                } else {
                    view.program.set_uniform4fm("model", Transform().data(), false);
                }

                view.draw();
                view.vao.unbind();
            }
        }
    }
}