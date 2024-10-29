//
// Created by alex on 16.07.24.
//

#include "PluginPicker.h"
#include "PickerGui.h"
#include "Engine.h"
#include "PluginGraphics.h"
#include "EventsCallbacks.h"
#include "Mouse.h"
#include "imgui.h"
#include "Transform.h"
#include "AABB.h"
//#include "KDTreeCpu.h"
#include "Cuda/KDTreeCuda.h"
#include "PointCloud.h"
#include "SurfaceMesh.h"
#include "EventsPicker.h"

namespace Bcg {

    static void on_construct_entity(entt::registry &registry, entt::entity entity_id) {
        Engine::Context().get<Picked>().entity.id = entity_id;
    }

    PluginPicker::PluginPicker() : Plugin("PluginPicker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }

        Engine::State().on_construct<entt::entity>().connect<&on_construct_entity>();
    }

    Picked &PluginPicker::pick(const ScreenSpacePos &pos) {
        auto &mouse = Engine::Context().get<Mouse>();
        auto &picked = last_picked();
        picked.spaces = mouse.cursor.last_left.press;
        picked.entity.is_background = picked.spaces.ndc.z == 1.0;
        auto view = Engine::State().view<AABB, Transform>();
        for (const auto entity_id: view) {
            auto &aabb = Engine::State().get<AABB>(entity_id);
            auto &transform = Engine::State().get<Transform>(entity_id);
            if (contains(aabb, (glm::inverse(transform.world()) * glm::vec4(picked.spaces.wsp, 1.0f)))) {
                picked.entity.id = entity_id;
                break;
            }
        }

        auto entity_id = picked.entity.id;
        if (Engine::valid(entity_id) && !picked.entity.is_background) {
            if (Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                picked.spaces.osp = glm::inverse(transform.world()) * glm::vec4(picked.spaces.wsp, 1.0f);
            }
/*            if (!Engine::has<KDTreeCpu>(entity_id)) {
                auto &kdtree = Engine::State().emplace<KDTreeCpu>(entity_id);
                if (Engine::has<SurfaceMesh>(entity_id)) {
                    auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
                    kdtree.build(mesh.positions());
                }else *//*if(Engine::has<Graph>(entity_id)){
                    auto &graph = Engine::State().get<Graph>(entity_id);
                    kdtree.build(graph.positions());
                }else *//*if (Engine::has<PointCloud>(entity_id)) {
                    auto &pc = Engine::State().get<PointCloud>(entity_id);
                    kdtree.build(pc.positions());
                }
            }

            auto &kdtree = Engine::State().get<KDTreeCpu>(entity_id);*/

            auto kdtree = cuda::KDTreeCuda(entity_id);
            if(!kdtree){
                if (Engine::has<SurfaceMesh>(entity_id)) {
                    auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
                    kdtree.build(mesh.positions());
                }else /*if(Engine::has<Graph>(entity_id)){
                    auto &graph = Engine::State().get<Graph>(entity_id);
                    kdtree.build(graph.positions());
                }else */if (Engine::has<PointCloud>(entity_id)) {
                    auto &pc = Engine::State().get<PointCloud>(entity_id);
                    kdtree.build(pc.positions());
                }
            }

            auto result = kdtree.radius_query(picked.spaces.osp, picked.entity.pick_radius);
            if (!result.indices.empty()) {
                picked.entity.vertex_idx = result.indices[0];
                Engine::Dispatcher().trigger(Events::PickedVertex{entity_id, &result.indices});
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
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
        Engine::Dispatcher().sink<Events::Callback::MouseButton>().connect<&on_mouse_button>();
        Plugin::activate();
    }

    void PluginPicker::begin_frame() {}

    void PluginPicker::update() {}

    void PluginPicker::end_frame() {}

    void PluginPicker::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::MouseButton>().disconnect<&on_mouse_button>();
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginPicker::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginPicker::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = last_picked();
                Gui::Show(picked);
                ImGui::End();
            }
        }
    }

    void PluginPicker::render() {

    }
}