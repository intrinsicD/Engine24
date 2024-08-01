//
// Created by alex on 31.07.24.
//

#include "PluginViews.h"
#include "Logger.h"
#include "imgui.h"
#include "Picker.h"
#include "Engine.h"
#include "Views.h"
#include "Mesh.h"
#include "PointCloud.h"
#include "FileWatcher.h"


namespace Bcg {
    void PluginViews::activate() {
        Plugin::activate();
    }

    void PluginViews::begin_frame() {}

    void PluginViews::update() {
        if(Engine::Context().find<FileWatcher>()){
            auto &watcher = Engine::Context().get<FileWatcher>();
            watcher.check();
        }
    }

    void PluginViews::end_frame() {}

    void PluginViews::deactivate() {
        Plugin::deactivate();
    }

    static bool show_views = false;

    void PluginViews::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("Views", nullptr, &show_views);
            ImGui::EndMenu();
        }
    }

    void PluginViews::render_gui() {
        if (show_views) {
            if (ImGui::Begin(name, &show_views, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                auto entity_id = picked.entity.id;

                if (!Engine::has<PointCloudView>(entity_id) && (Engine::has<PointCloud>(entity_id) ||
                                                                /*Engine::has<Graph>(entity_id) ||*/
                                                                Engine::has<SurfaceMesh>(entity_id))) {
                    if (ImGui::Button("Make Points Visible")) {
                        //trigger the respective Command
                    }
                }

                if (!Engine::has<GraphView>(entity_id) &&
                    (/*Engine::has<Graph>(entity_id) ||*/ Engine::has<SurfaceMesh>(entity_id))) {
                    if (ImGui::Button("Make Graph Visible")) {
                        //trigger the respective Command
                    }
                }

                if (!Engine::has<MeshView>(entity_id) && Engine::has<SurfaceMesh>(entity_id)) {
                    if (ImGui::Button("Make Mesh Visible")) {
                        //trigger the respective Command
                    }
                }

                if (Engine::has<VectorfieldView>(entity_id)) {

                }

                if (Engine::has<VectorfieldViews>(entity_id)) {
                    auto &vectorfield_views = Engine::State().get<VectorfieldViews>(entity_id);
                    ImGui::Checkbox("Hide##vectorfield_views", &vectorfield_views.hide);
                    for (auto &item: vectorfield_views) {

                    }
                }
            }
            ImGui::End();
        }
    }

    void PluginViews::render() {}
}