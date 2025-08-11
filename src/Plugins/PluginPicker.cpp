//
// Created by alex on 16.07.24.
//

#include "PluginPicker.h"
#include "Engine.h"
#include "../../Graphics/ModuleGraphics.h"
#include "PluginSelection.h"
#include "EventsCallbacks.h"
#include "Mouse.h"
#include "imgui.h"
#include "WorldTransformComponent.h"
#include "ModuleAABB.h"
#include "ModuleMesh.h"
#include "Cuda/BVHCudaNew.h"
#include "PointCloud.h"
#include "GetPrimitives.h"
#include "EventsPicker.h"
#include "Entity.h"

namespace Bcg {
    static void on_construct_entity(entt::registry &registry, entt::entity entity_id) {
        Engine::Context().get<Picked>().entity.id = entity_id;
    }

    PluginPicker::PluginPicker() : Plugin("PluginPicker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
    }

    Picked &PluginPicker::pick(const ScreenSpacePos &pos) {
        auto &mouse = Engine::Context().get<Mouse>();
        auto &picked = last_picked();
        picked.spaces = mouse.cursor.last_left.press;
        picked.entity.is_background = picked.spaces.ndc.z == 1.0;
        auto view = Engine::State().view<AABBHandle, WorldTransformComponent>();
        for (const auto entity_id: view) {
            auto h_aabb = ModuleAABB::get(entity_id);
            auto &transform = Engine::State().get<WorldTransformComponent>(entity_id);
            if (AABBUtils::Contains(*h_aabb, (glm::inverse(transform.world_transform) * glm::vec4(picked.spaces.wsp, 1.0f)))) {
                picked.entity.id = entity_id;
                break;
            }
        }

        auto entity_id = picked.entity.id;
        if (Engine::valid(entity_id) && !picked.entity.is_background) {
            if (Engine::has<WorldTransformComponent>(entity_id)) {
                auto &transform = Engine::State().get<WorldTransformComponent>(entity_id);
                picked.spaces.osp = glm::inverse(transform.world_transform) * glm::vec4(picked.spaces.wsp, 1.0f);
            }

            auto kdtree = cuda::BVHCudaNew(entity_id);
            if (!kdtree) {
                auto *vertices = GetPrimitives(entity_id).vertices();
                if (vertices) {
                    auto positions = vertices->get<Vector<float, 3> >("v:position");
                    kdtree.build(positions.vector());
                } else {
                    Log::Error("PluginPicker::pick: Entity {} does not have vertices.", entity_id);
                    return picked;
                }
            }

            auto result = kdtree.radius_query(picked.spaces.osp, picked.entity.pick_radius);
            if (!result.empty()) {
                picked.entity.vertex_idx = result.indices[0];
                auto indices = std::vector<size_t>(result.indices.data(),
                                                   result.indices.data() + result.indices.size());
                Engine::Dispatcher().trigger(Events::PickedVertex{entity_id, &indices});
            }
            Engine::Dispatcher().trigger(Events::PickedEntity{entity_id});
        } else {
            Engine::Dispatcher().trigger<Events::PickedBackgound>();
        }

        return picked;
    }

    Picked &PluginPicker::last_picked() {
        return Engine::Context().get<Picked>();
    }

    static void on_mouse_button(const Events::Callback::MouseButton &event) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (event.action) {
            PluginPicker::pick(mouse.cursor.current.ssp);
        }
    }


    void PluginPicker::activate() {
        if (base_activate()) {
            if (!Engine::Context().find<Picked>()) {
                Engine::Context().emplace<Picked>();
            }
            Engine::Dispatcher().sink<Events::Callback::MouseButton>().connect<&on_mouse_button>();
            Engine::State().on_construct<entt::entity>().connect<&on_construct_entity>();
        }
    }

    void PluginPicker::begin_frame() {
    }

    void PluginPicker::update() {
    }

    void PluginPicker::end_frame() {
    }

    void PluginPicker::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::Callback::MouseButton>().disconnect<&on_mouse_button>();
            Engine::State().on_construct<entt::entity>().disconnect<&on_construct_entity>();
        }
    }

    static bool gui_enabled = false;

    void PluginPicker::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem(name.c_str(), nullptr, &gui_enabled);
            ImGui::EndMenu();
        }
    }

    void PluginPicker::render_gui() {
        if (gui_enabled) {
            if (ImGui::Begin(name.c_str(), &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = last_picked();
                show_gui(picked);
                ImGui::End();
            }
        }
    }

    void PluginPicker::render() {
    }

    void PluginPicker::show_gui(Picked &picked) {
        auto &entity = picked.entity;
        ImGui::Text("entity id: %u", static_cast<entt::id_type>(entity.id));
        ImGui::Text("is_background: %s", entity.is_background ? "true" : "false");
        ImGui::Text("vertex_idx: %u", entity.vertex_idx);
        ImGui::Text("edge_idx: %u", entity.edge_idx);
        ImGui::Text("face_idx: %u", entity.face_idx);
        ImGui::DragScalar("pick_radius",
                          ImGuiDataType_Float,
                          &entity.pick_radius,
                          0.01f,
                          nullptr,
                          nullptr,
                          "%.2f",
                          ImGuiSliderFlags_AlwaysClamp);

        if (ImGui::CollapsingHeader("Spaces")) {
            show_gui(picked.spaces);
        }
    }

    void PluginPicker::show_gui(const Points &spaces) {
        ImGui::Text("Sceen Space Position:               (%f, %f)", spaces.ssp.x, spaces.ssp.y);
        ImGui::Text("Screen Space Position Dpi Adjusted: (%f, %f)", spaces.sspda.x, spaces.sspda.y);
        ImGui::Text("Normalized Device Coordinates:      (%f, %f, %f)", spaces.ndc.x, spaces.ndc.y, spaces.ndc.z);
        ImGui::Text("View Space Position:                (%f, %f, %f)", spaces.vsp.x, spaces.vsp.y, spaces.vsp.z);
        ImGui::Text("World Space Position:               (%f, %f, %f)", spaces.wsp.x, spaces.wsp.y, spaces.wsp.z);
        ImGui::Text("Object Space Position:              (%f, %f, %f)", spaces.osp.x, spaces.osp.y, spaces.osp.z);
    }
}
