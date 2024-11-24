//
// Created by alex on 15.07.24.
//

#include "PluginAABB.h"
#include "Engine.h"
#include "Entity.h"
#include "AABBGui.h"
#include "Picker.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "EventsEntity.h"
#include "Types.h"
#include "PropertiesGui.h"
#include "BoundingVolumes.h"

namespace Bcg {
    static void on_cleanup_components(const Events::Entity::CleanupComponents &event) {
        Commands::Cleanup<AABB>(event.entity_id).execute();
    }

    PluginAABB::PluginAABB() : Plugin("AABB") {
    }

    void PluginAABB::activate() {
        Engine::Dispatcher().sink<Events::Entity::CleanupComponents>().connect<&on_cleanup_components>();
        Plugin::activate();
        if (!Engine::Context().find<Pool<AABB> >()) {
            auto &pool = Engine::Context().emplace<Pool<AABB> >();
            pool.create();
        }
    }

    void PluginAABB::begin_frame() {
    }

    void PluginAABB::update() {
    }

    void PluginAABB::end_frame() {
    }

    void PluginAABB::deactivate() {
        Engine::Dispatcher().sink<Events::Entity::CleanupComponents>().disconnect<&on_cleanup_components>();
        Plugin::deactivate();
    }

    static bool show_gui = false;
    static bool show_pool_gui = false;

    void PluginAABB::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            if(ImGui::BeginMenu(name)) {
                ImGui::MenuItem("Instance", nullptr, &show_gui);
                ImGui::MenuItem("Pool", nullptr, &show_pool_gui);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginAABB::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                Gui::Show(picked.entity.id);
            }
            ImGui::End();
        }
        if (show_pool_gui) {
            if (ImGui::Begin("Pool", &show_pool_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &pool = Engine::Context().get<Pool<AABB> >();
                Gui::Show("AABBPoolProperties",pool.properties);
            }
            ImGui::End();
        }
    }

    void PluginAABB::render() {
    }


    void Commands::Setup<AABB>::execute() const {
        if (!Engine::valid(entity_id)) {
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("{} failed, entity {} has no vertices.", name, entity_id);
            return;
        }

        auto positions = vertices->get<PointType>("v:position");
        if (!positions) {
            Log::Warn("{} failed, entity {} has no property {}.", name, entity_id, positions.name());
            return;
        }

        auto &pool = Engine::Context().get<Pool<AABB> >();
        auto &bv = Engine::State().get_or_emplace<BoundingVolumes>(entity_id);
        bv.h_aabb = pool.create();
        *bv.h_aabb = AABB::Build(positions.vector().begin(), positions.vector().end());
        Log::Info("{} for entity {}", name, entity_id);
    }

    void Commands::Cleanup<AABB>::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<BoundingVolumes>(entity_id) ||
            Engine::State().get<BoundingVolumes>(entity_id).h_aabb.is_valid()) {
            Log::Warn(name + "Entity does not have an AABB. Abort Command");
            return;
        }

        auto &bv = Engine::State().get<BoundingVolumes>(entity_id);

        if (bv.h_aabb.is_valid()) {
            auto &pool = Engine::Context().get<Pool<AABB> >();
            pool.destroy(bv.h_aabb);
            assert(!bv.h_aabb.is_valid());
        }

        Log::Info("{} for entity {}", name, entity_id);
    }

    void Commands::CenterAndScaleByAABB::execute() const {
        if (!Engine::valid(entity_id)) {
            return;
        }

        if (!Engine::has<BoundingVolumes>(entity_id)) {
            return;
        }

        auto &bv = Engine::State().get<BoundingVolumes>(entity_id);
        if (!bv.h_aabb.is_valid()) {
            return;
        }

        auto &aabb = *bv.h_aabb;

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("{} failed, entity {} has no vertices.", name, entity_id);
            return;
        }

        auto data = vertices->get<PointType>(property_name);
        if (!data) {
            Log::Warn("{} failed, entity {} has no property {}.", name, entity_id, property_name);
            return;
        }

        Vector<float, 3> c = aabb.center();
        float s = glm::compMax(aabb.max - aabb.min);

        for (auto &point: data.vector()) {
            point -= c;
            point /= s;
        }

        aabb.min = (aabb.min - c) / s;
        aabb.max = (aabb.max - c) / s;
    }
}
