//
// Created by alex on 15.07.24.
//

#include "ModuleAABB.h"
#include "Engine.h"
#include "Entity.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "EventsEntity.h"
#include "Types.h"
#include "Picker.h"

namespace Bcg {
    ModuleAABB::ModuleAABB() : Module("AABB") {

    }

    void ModuleAABB::activate() {
        if (base_activate()) {
            if (!Engine::Context().find<AABBPool>()) {
                Engine::Context().emplace<AABBPool>();
            }
        }
    }

    void ModuleAABB::deactivate() {
        if (base_deactivate()) {
            if (Engine::Context().find<AABBPool>()) {
                Engine::Context().erase<AABBPool>();
            }
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

    void ModuleAABB::show_gui(const PoolHandle<AABB<float>> &h_aabb) {
        if (h_aabb.is_valid()) {
            show_gui(*h_aabb);
        }
    }

    void ModuleAABB::show_gui(const AABB<float> &aabb) {
        ImGui::Text("Min: (%f, %f, %f)", aabb.min.x, aabb.min.y, aabb.min.z);
        ImGui::Text("Max: (%f, %f, %f)", aabb.max.x, aabb.max.y, aabb.max.z);
    }

    void ModuleAABB::show_gui(entt::entity entity_id) {
        if (has(entity_id)) {
            show_gui(get(entity_id));
        }
    }

    AABBHandle ModuleAABB::make_handle(const AABB<float> &object) {
        auto &pool = Engine::Context().get<AABBPool>();
        return pool.create_smart_handle(object);
    }

    AABBHandle ModuleAABB::create(entt::entity entity_id, const AABB<float> &object) {
        auto handle = make_handle(object);
        return add(entity_id, handle);
    }

    AABBHandle ModuleAABB::add(entt::entity entity_id, AABBHandle h_object) {
        return Engine::State().get_or_emplace<AABBHandle>(entity_id, h_object);
    }

    void ModuleAABB::remove(entt::entity entity_id) {
        Engine::State().remove<AABBHandle>(entity_id);
    }

    bool ModuleAABB::has(entt::entity entity_id) {
        return Engine::State().all_of<AABBHandle>(entity_id);
    }

    AABBHandle ModuleAABB::get(entt::entity entity_id) {
        return Engine::State().get<AABBHandle>(entity_id);
    }

    static std::string s_name = "AABB";

    void ModuleAABB::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("Setup {} failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("Setup {} failed, entity {} has no vertices.", s_name, entity_id);
            return;
        }

        auto positions = vertices->get<PointType>("v:position");
        if (!positions) {
            Log::Warn("Setup {} failed, entity {} has no property {}.", s_name, entity_id, positions.name());
            return;
        }

        ModuleAABB::create(entity_id, AABB<float>::Build(positions.vector().begin(), positions.vector().end()));
        Log::Info("Setup {} for entity {}", s_name, entity_id);
    }

    void ModuleAABB::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("Cleanup {} failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        if (!Engine::has<AABBHandle>(entity_id)) {
            Log::Warn("Cleanup {} failed, Entity {} does not have an {}. Abort Command", s_name, entity_id, s_name);
            return;
        }

        ModuleAABB::remove(entity_id);

        Log::Info("Cleanup {} for entity {}", s_name, entity_id);
    }

    void ModuleAABB::center_and_scale_by_aabb(entt::entity entity_id, const std::string &property_name) {
        if (!Engine::valid(entity_id)) {
            return;
        }

        if (!Engine::has<AABBHandle>(entity_id)) {
            return;
        }

        auto h_aabb = Engine::State().get<AABBHandle>(entity_id);
        if (!h_aabb.is_valid()) {
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("Center and scale by {} failed, entity {} has no vertices.", s_name, entity_id);
            return;
        }

        auto data = vertices->get<PointType>(property_name);
        if (!data) {
            Log::Warn("Center and scale by {} failed, entity {} has no property {}.", s_name, entity_id, property_name);
            return;
        }

        Vector<float, 3> c = h_aabb->center();
        float s = glm::compMax(h_aabb->max - h_aabb->min);

        for (auto &point: data.vector()) {
            point -= c;
            point /= s;
        }

        h_aabb->min = (h_aabb->min - c) / s;
        h_aabb->max = (h_aabb->max - c) / s;
        Log::Info("Center and scale by {} for entity {}", s_name, entity_id);
    }

}
