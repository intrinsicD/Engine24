//
// Created by alex on 15.07.24.
//

#include "PluginTransform.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "Engine.h"
#include "EventsGui.h"
#include "Picker.h"
#include "TransformGui.h"
#include "../Camera/Camera.h"
#include "Graphics.h"

namespace Bcg {
    PluginTransform::PluginTransform() : Plugin("Transform") {}

    void PluginTransform::activate() {
        Plugin::activate();
    }

    void PluginTransform::begin_frame() {

    }

    void PluginTransform::update() {

    }

    void PluginTransform::end_frame() {

    }

    void PluginTransform::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    static void on_gui_render(const Events::Gui::Render &event) {
        if (!show_gui) {
            Engine::Dispatcher().sink<Events::Gui::Render>().disconnect<on_gui_render>();
            return;
        }

        auto &picked = Engine::Context().get<Picked>();
        Gui::ShowTransform(picked.entity.id, show_gui);
    }

    void PluginTransform::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::MenuItem(name, nullptr, &show_gui)) {
                Engine::Dispatcher().sink<Events::Gui::Render>().connect<on_gui_render>();
            }
            ImGui::EndMenu();
        }
    }

    void PluginTransform::render_gui() {


    }

    void PluginTransform::render() {

    }
}