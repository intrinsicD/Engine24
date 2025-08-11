//
// Created by alex on 02.06.25.
//

#include "GuiModuleCamera.h"
#include "CameraGui.h"
#include "Engine.h"
#include "imgui.h"

namespace Bcg{
    GuiModuleCamera::GuiModuleCamera() : GuiModule("Camera") {
    }

    void GuiModuleCamera::activate() {
        if (base_activate()) {

        }
    }

    void GuiModuleCamera::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool show_gui = false;

    void GuiModuleCamera::render_menu() {
        if (ImGui::BeginMenu("Rendering")) {
            ImGui::MenuItem(name.c_str(), nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void GuiModuleCamera::render_gui() {
        // Implement GUI rendering logic here
        if (show_gui) {
            if (ImGui::Begin(name.c_str(), &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::Show(Engine::Context().get<Camera>());
                ImGui::End();
            }
        }
    }
}