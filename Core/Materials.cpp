//
// Created by alex on 03.07.24.
//

#include "Materials.h"
#include "imgui.h"
#include "Engine.h"
#include "Picker.h"
#include "Material.h"
#include "Graphics.h"

namespace Bcg {
    static bool show_gui = false;

    Materials::Materials() : Plugin("Materials") {

    }

    void Materials::setup(entt::entity entity_id, MeshMaterial &material) {
        auto v_position = Graphics::get_or_add_buffer("v_position");
    }

    void Materials::setup(entt::entity entity_id, GraphMaterial &material) {

    }

    void Materials::setup(entt::entity entity_id, PointCloudMaterial &material) {

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
                        Gui::Show(Engine::State().get<PointCloudMaterial>(picker.entity.id));
                    }
                    if (Engine::State().all_of<GraphMaterial>(picker.entity.id)) {
                        Gui::Show(Engine::State().get<GraphMaterial>(picker.entity.id));
                    }
                    if (Engine::State().all_of<MeshMaterial>(picker.entity.id)) {
                        Gui::Show(Engine::State().get<MeshMaterial>(picker.entity.id));
                    }
                }

            }
            ImGui::End();
        }
    }

    void Materials::render() {

    }
}