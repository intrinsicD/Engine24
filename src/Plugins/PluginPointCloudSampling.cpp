//
// Created by alex on 11/3/24.
//

#include "PluginPointCloudSampling.h"
#include "Engine.h"
#include "imgui.h"
#include "Picker.h"

namespace Bcg {
    PluginPointCloudSampling::PluginPointCloudSampling() : Plugin("PointCloudSampling") {
    }

    void PluginPointCloudSampling::activate() {
        if (base_activate()) {

        }
    }

    void PluginPointCloudSampling::begin_frame() {
    }

    void PluginPointCloudSampling::update() {
    }

    void PluginPointCloudSampling::end_frame() {
    }

    void PluginPointCloudSampling::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool show_gui = false;

    void PluginPointCloudSampling::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            if (ImGui::BeginMenu("PointCloud")) {
                ImGui::MenuItem("Sampling", nullptr, &show_gui);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginPointCloudSampling::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name.c_str(), &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                ImGui::Text("ToDo Implement");
            }
            ImGui::End();
        }
    }

    void PluginPointCloudSampling::render() {
    }
}
