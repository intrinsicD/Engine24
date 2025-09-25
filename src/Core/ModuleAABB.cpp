//
// Created by alex on 15.07.24.
//

#include "ModuleAABB.h"
#include "Engine.h"
#include "Entity.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "AABBComponents.h"
#include "Types.h"
#include "Picker.h"

namespace Bcg {
    ModuleAABB::ModuleAABB() : Module("AABB") {

    }

    void ModuleAABB::activate() {
        if (base_activate()) {

        }
    }

    void ModuleAABB::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool gui_enabled = false;

    void ModuleAABB::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem("AABB", nullptr, &gui_enabled);
            ImGui::EndMenu();
        }
    }

    void ModuleAABB::render_gui() {
        if (gui_enabled) {
            if (ImGui::Begin("AABB", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void ModuleAABB::show_gui(const char *label, const AABB<float> &aabb) {
        ImGui::Text("%s", label);
        ImGui::Text("Min: (%f, %f, %f)", aabb.min.x, aabb.min.y, aabb.min.z);
        ImGui::Text("Max: (%f, %f, %f)", aabb.max.x, aabb.max.y, aabb.max.z);
        const auto center = aabb.center();
        ImGui::Text("Center: (%f, %f, %f)", center.x, center.y, center.z);
        const auto diagonal = aabb.diagonal();
        ImGui::Text("Diagonal: (%f, %f, %f)", diagonal.x, diagonal.y, diagonal.z);
        const auto extent = diagonal / 2.0f;
        ImGui::Text("Extent: (%f, %f, %f)", extent.x, extent.y, extent.z);
    }

    void ModuleAABB::show_gui(entt::entity entity_id) {
        if (Engine::has<LocalAABB>(entity_id)) {
            show_gui("local_aabb", Engine::State().get<LocalAABB>(entity_id).aabb);
            ImGui::Separator();
        }
        if (Engine::has<WorldAABB>(entity_id)) {
            if (Engine::has<LocalAABB>(entity_id)) {
                ImGui::Separator();
            }
            show_gui("world_aabb", Engine::State().get<WorldAABB>(entity_id).aabb);
        }
    }


    static std::string s_name = "AABB";

    void ModuleAABB::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("Setup {} failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("Setup Local {} failed, entity {} has no vertices.", s_name, entity_id);
            return;
        }

        auto positions = vertices->get<PointType>("v:point");
        if (!positions) {
            Log::Warn("Setup Local {} failed, entity {} has no property {}.", s_name, entity_id, positions.name());
            return;
        }
        Engine::State().emplace<LocalAABB>(entity_id, AABB<float>::Build(positions.vector().begin(), positions.vector().end()));
        Log::Info("Setup Local {} for entity {}", s_name, entity_id);
    }

    void ModuleAABB::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("Cleanup {} failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        Engine::State().remove<LocalAABB>(entity_id);
        Engine::State().remove<WorldAABB>(entity_id);
    }
}
