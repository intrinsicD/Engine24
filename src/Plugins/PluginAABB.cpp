//
// Created by alex on 15.07.24.
//

#include "PluginAABB.h"
#include "Engine.h"
#include "AABBGui.h"
#include "Picker.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "EventsEntity.h"
#include "Types.h"

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

    void PluginAABB::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
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
    }

    void PluginAABB::render() {

    }

    namespace Commands {
        void Setup<AABB>::execute() const {
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

            auto &aabb = Engine::require<AABB>(entity_id);
            Engine::Dispatcher().trigger(Events::Entity::PreAdd<AABB>{entity_id});
            Build(aabb, positions.vector());
            Engine::Dispatcher().trigger(Events::Entity::PostAdd<AABB>{entity_id});
            Log::Info("{} for entity {}", name, entity_id);
        }

        void Cleanup<AABB>::execute() const {
            if (!Engine::valid(entity_id)) {
                return;
            }

            if (Engine::has<AABB>(entity_id)) {
                Engine::Dispatcher().trigger(Events::Entity::PreRemove<AABB>{entity_id});
                Engine::State().remove<AABB>(entity_id);
                Engine::Dispatcher().trigger(Events::Entity::PostRemove<AABB>{entity_id});
                Log::Info("Cleanup<AABB> for entity {}", entity_id);
            }
        }

        void CenterAndScaleByAABB::execute() const {
            if (!Engine::valid(entity_id)) {
                return;
            }

            if (!Engine::has<AABB>(entity_id)) {
                return;
            }

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

            auto &aabb = Engine::require<AABB>(entity_id);

            Vector<float, 3> center = aabb.center();
            float scale = (aabb.max - aabb.min).maxCoeff();

            for (auto &point: data.vector()) {
                point -= center;
                point /= scale;
            }

            aabb.min = (aabb.min - center) / scale;
            aabb.max = (aabb.max - center) / scale;
        }
    }
}