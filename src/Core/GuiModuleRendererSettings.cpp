//
// Created by alex on 16.06.25.
//

#include "GuiModuleRendererSettings.h"
#include "imgui.h"

namespace Bcg {
    GuiModuleRendererSettings::GuiModuleRendererSettings(Renderer &renderer) : GuiModule("Renderer Settings"), m_renderer(renderer) {

    }

    void GuiModuleRendererSettings::render_menu() {
        if (ImGui::BeginMenu("Engine")) {
            ImGui::MenuItem(name.c_str(), nullptr, &m_is_window_open);
            ImGui::EndMenu();
        }
    }

    void GuiModuleRendererSettings::render_gui() {
        if (!m_is_window_open) {
            return;
        }

        if (ImGui::Begin(name.c_str(), &m_is_window_open)) {
            ImGui::Text("Background Color");
            ImGui::Separator();

            Vector<float, 4> clear_color = m_renderer.get_clear_color();
            if (ImGui::ColorEdit4("Clear Color", &clear_color[0])) {
                m_renderer.set_clear_color(clear_color);
            }
        }
        ImGui::End();
    }
}