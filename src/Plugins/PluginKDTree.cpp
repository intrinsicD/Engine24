//
// Created by alex on 01.08.24.
//

#include "PluginKDTree.h"
#include "Engine.h"
#include "imgui.h"
#include "Logger.h"
#include "KDTreeCpu.h"
#include "Picker.h"
#include "Cuda/BVHCuda.h"
#include "SurfaceMesh.h"
#include "PointCloud.h"
#include "KDTreeGui.h"

namespace Bcg {
    static void on_construct_entity(entt::registry &registry, entt::entity entity_id) {
        auto kdtree = cuda::BVHCuda(entity_id);
        if (!kdtree) {
            if (Engine::has<SurfaceMesh>(entity_id)) {
                auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
                kdtree.build(mesh.positions());
            } else /*if(Engine::has<Graph>(entity_id)){
                    auto &graph = Engine::State().get<Graph>(entity_id);
                    kdtree.build(graph.positions());
                }else */if (Engine::has<PointCloud>(entity_id)) {
                auto &pc = Engine::State().get<PointCloud>(entity_id);
                kdtree.build(pc.positions());
            }
        }
    }

    PluginKDTree::PluginKDTree() : Plugin("KDTree") {

    }

    void PluginKDTree::activate() {
        if (base_activate()) {
            Engine::State().on_construct<entt::entity>().connect<&on_construct_entity>();
        }

    }

    void PluginKDTree::begin_frame() {}

    void PluginKDTree::update() {}

    void PluginKDTree::end_frame() {}

    void PluginKDTree::deactivate() {
        if (base_deactivate()) {
            Engine::State().on_construct<entt::entity>().disconnect<&on_construct_entity>();
        }
    }

    static bool show_gui = false;

    void PluginKDTree::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            ImGui::MenuItem(name.c_str(), nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginKDTree::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name.c_str(), &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picker = Engine::Context().get<Picked>();
                if(Engine::valid(picker.entity.id)){
                    Gui::ShowKDTree(picker.entity.id);
                }
                ImGui::End();
            }
        }
    }

    void PluginKDTree::render() {}

}