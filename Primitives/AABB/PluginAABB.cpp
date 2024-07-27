//
// Created by alex on 15.07.24.
//

#include "PluginAABB.h"
#include "Engine.h"
#include "AABBGui.h"
#include "Picker.h"
#include "imgui.h"

namespace Bcg {
    PluginAABB::PluginAABB() : Plugin("AABB") {}

    void PluginAABB::activate() {
        Plugin::activate();
    }

    void PluginAABB::begin_frame() {

    }

    void PluginAABB::update() {

    }

    void PluginAABB::end_frame() {

    }

    void PluginAABB::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;
    static bool show_pool = false;

    void PluginAABB::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::MenuItem("Pool##AABB", nullptr, &show_pool);
            ImGui::EndMenu();
        }
    }

    void PluginAABB::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("AABB", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::Show(picked.entity.id);
            }
            ImGui::End();
        }
        if (show_pool) {
            auto &pool = Engine::Context().get<AABBPool>();
            if (ImGui::Begin("AABB Pool", &show_pool, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::Show(pool);
            }
            ImGui::End();
        }
    }

    void PluginAABB::render() {

    }
}