//
// Created by alex on 03.07.24.
//

#include "Materials.h"
#include "imgui.h"
#include "Engine.h"
#include "Picker.h"

namespace Bcg {
    static bool show_gui = false;

    Materials::Materials() : Plugin("Materials") {

    }

    void Materials::render_gui(const Material &material) {
        ImGui::Text("vao: %u", material.vao);
        ImGui::Text("program: %u", material.program);
        ImGui::Text("offset: %u", material.offset);
        ImGui::Text("size: %u", material.size);

        if (ImGui::CollapsingHeader("Textures")) {
            for (auto &item: material.textures) {
                ImGui::Text("%s: %u", item.first.c_str(), item.second);
            }
        }
    }

    void Materials::render_gui(MeshMaterial &material) {
        ImGui::PushID("MeshMaterial");
        ImGui::ColorEdit3("BaseColor", &material.base_color[0]);
        if (ImGui::CollapsingHeader("Base")) {
            render_gui(static_cast<const Material &>(material));
        }
        ImGui::PopID();
    }

    void Materials::render_gui(GraphMaterial &material) {
        ImGui::PushID("GraphMaterial");
        if (ImGui::CollapsingHeader("Base")) {
            render_gui(static_cast<const Material &>(material));
        }
        ImGui::PopID();
    }

    void Materials::render_gui(PointCloudMaterial &material) {
        ImGui::PushID("PointCloudMaterial");
        if (ImGui::CollapsingHeader("Base")) {
            render_gui(static_cast<const Material &>(material));
        }
        ImGui::PopID();
    }

    void Materials::activate() {
        Plugin::activate();
    }

    void Materials::begin_frame() {

    }

    void Materials::update() {

    }

    void Materials::end_frame() {

    }

    void Materials::deactivate() {
        Plugin::deactivate();
    }

    void Materials::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("Material", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void Materials::render_gui() {
        if (show_gui) {
            if (ImGui::Begin("Material", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picker = Picker::last_picked();
                if (picker.entity) {
                    if (Engine::State().all_of<PointCloudMaterial>(picker.entity.id)) {
                        render_gui(Engine::State().get<PointCloudMaterial>(picker.entity.id));
                    }
                    if (Engine::State().all_of<GraphMaterial>(picker.entity.id)) {
                        render_gui(Engine::State().get<GraphMaterial>(picker.entity.id));
                    }
                    if (Engine::State().all_of<MeshMaterial>(picker.entity.id)) {
                        render_gui(Engine::State().get<MeshMaterial>(picker.entity.id));
                    }
                }

            }
            ImGui::End();
        }
    }

    void Materials::render() {

    }
}