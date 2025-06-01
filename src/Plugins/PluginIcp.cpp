//
// Created by alex on 13.08.24.
//

#include "PluginIcp.h"
#include "imgui.h"
#include "GuiUtils.h"

namespace Bcg {
    PluginIcp::PluginIcp() : Plugin("ICP") {

    }

    void PluginIcp::activate() {
        if (base_activate()) {

        }
    }

    void PluginIcp::begin_frame() {

    }

    void PluginIcp::update() {

    }

    void PluginIcp::end_frame() {

    }

    void PluginIcp::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool show_gui = false;

    void PluginIcp::render_menu() {
        if (ImGui::BeginMenu("Registration")) {
            ImGui::MenuItem("ICP", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginIcp::render_gui() {
        if (show_gui) {
            if (ImGui::Begin("ICP", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                static std::pair<entt::entity, std::string> source;
                static std::pair<entt::entity, std::string> target;
                bool changed = Gui::ComboEntities("Source", source);
                ImGui::SameLine();
                changed |= Gui::ComboEntities("Target", target);



            }
            ImGui::End();
        }
    }

    void PluginIcp::render() {

    }
}