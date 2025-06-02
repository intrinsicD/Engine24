//
// Created by alex on 02.06.25.
//

#include "GuiModuleTransform.h"
#include "TransformUtils.h"
#include "TransformGui.h"
#include "Engine.h"
#include "Picker.h"
#include "imgui.h"

namespace Bcg{
    GuiModuleTransform::GuiModuleTransform() : GuiModule("Transform") {
    }

    void GuiModuleTransform::activate() {
        if (base_activate()) {

        }
    }

    void GuiModuleTransform::begin_frame() {

    }

    void GuiModuleTransform::update() {

    }

    void GuiModuleTransform::end_frame() {

    }

    void GuiModuleTransform::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool show_gui = false;

    void GuiModuleTransform::render_menu() {
        if (ImGui::BeginMenu("Transform")) {
            ImGui::MenuItem(name.c_str(), nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void GuiModuleTransform::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name.c_str(), &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                if (picked.entity) {
                    Gui::ShowTransform(picked.entity.id);
                } else {
                    ImGui::Text("No entity selected");
                }
                ImGui::End();
            }
        }
    }
}