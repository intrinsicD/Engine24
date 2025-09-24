//
// Created by alex on 12.08.25.
//

#include "ModuleGraph.h"

#include "ModuleCamera.h"
#include "TransformComponent.h"
#include "TransformUtils.h"
#include "ModuleAABB.h"

#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "PropertiesGui.h"

#include "Engine.h"
#include "Picker.h"
#include "GeometryUtils.h"

#include "ModuleGraphView.h"

namespace Bcg {
    ModuleGraph::ModuleGraph() : Module("GraphModule") {
    }

    void ModuleGraph::activate() {
        if (base_activate()) {

        }
    }

    void ModuleGraph::deactivate() {
        if (base_deactivate()) {

        }
    }

    static std::string s_name = "GraphModule";

    void ModuleGraph::remove(entt::entity entity_id) {
        Engine::State().remove<GraphInterface>(entity_id);
    }

    bool ModuleGraph::has(entt::entity entity_id) {
        return Engine::State().all_of<GraphInterface>(entity_id);
    }

    void ModuleGraph::destroy_entity(entt::entity entity_id) {
        remove(entity_id);
        if (Engine::State().all_of<Vertices>(entity_id)) {
            Engine::State().remove<Vertices>(entity_id);
        }
    }


    template<>
    struct BuilderTraits<AABB<float>, GraphInterface> {
        static AABB<float> build(const GraphInterface &gi) noexcept {
            return AABB<float>::Build(gi.vpoint.vector().begin(), gi.vpoint.vector().end());
        }
    };

    void ModuleGraph::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Setup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        auto &gi = Require<GraphInterface>(entity_id, Engine::State());

        auto h_aabb = ModuleAABB::create(entity_id, BuilderTraits<AABB<float>, GraphInterface>::build(gi));

        auto &transform = Engine::require<TransformComponent>(entity_id);

        //ScaleAndCenterAt(transform, h_aabb->center(), glm::compMax(h_aabb->diagonal()));

        /*
        ModuleAABB::center_and_scale_by_aabb(entity_id, gi.vpoint.name());
        ModuleCamera::center_camera_at_distance(h_aabb->center(), glm::compMax(h_aabb->diagonal()));
        */

        ModuleGraphView::setup(entity_id);
        Log::Info("#v: {}", gi.vertices.n_vertices());
        Log::Info("#h: {}", gi.halfedges.n_halfedges());
        Log::Info("#e: {}", gi.edges.n_edges());
    }

    void ModuleGraph::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Cleanup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        if (!Engine::State().all_of<GraphInterface>(entity_id)) {
            Log::Warn("{}::Cleanup failed, Entity {} does not have a GraphHandle. Abort Command", s_name,
                      static_cast<int>(entity_id));
            return;
        }

        remove(entity_id);
    }

    static bool gui_enabled = false;


    void ModuleGraph::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem("Graph Info", nullptr, &gui_enabled);
            ImGui::EndMenu();
        }
    }
    

    void ModuleGraph::render_gui() {
        if (gui_enabled) {
            if (ImGui::Begin("Graph Info", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void ModuleGraph::show_gui(const GraphInterface &gi) {
        if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(gi.vertices.n_vertices())).c_str())) {
            ImGui::PushID("Vertices");
            Gui::Show("##Vertices", gi.vertices);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Halfedges #v: " + std::to_string(gi.halfedges.n_halfedges())).c_str())) {
            ImGui::PushID("Halfedges");
            Gui::Show("##Halfedges", gi.halfedges);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Edges #v: " + std::to_string(gi.edges.n_edges())).c_str())) {
            ImGui::PushID("Edges");
            Gui::Show("##Edges", gi.edges);
            ImGui::PopID();
        }
    }

    void ModuleGraph::show_gui(entt::entity entity_id) {
        if (has(entity_id)) {
            show_gui(Require<GraphInterface>(entity_id, Engine::State()));
        }
    }
}
