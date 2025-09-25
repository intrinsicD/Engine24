//
// Created by alex on 12.08.25.
//

#include "ModuleGraph.h"

#include "ModuleCamera.h"
#include "TransformComponent.h"
#include "TransformUtils.h"
#include "AABBComponents.h"
#include "CameraUtils.h"

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


    void ModuleGraph::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Setup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        auto &gi = Require<GraphInterface>(entity_id, Engine::State());

        if (!Engine::has<LocalAABB>(entity_id)) {
            auto &local = Engine::require<LocalAABB>(entity_id);
            local.aabb = BuilderTraits<AABB<float>, std::vector<Vector<float, 3>>>::build(gi.vpoint.vector());
        }

        auto &local = Engine::require<LocalAABB>(entity_id);

        if (!Engine::has<TransformComponent>(entity_id)) {
            const auto &camera = Engine::Context().get<Camera>();
            const auto view_params = GetViewParams(camera);

            // Robust target scale based on bounding "radius"
            const glm::vec3 half_extent = local.aabb.half_extent();
            const float max_extent = glm::compMax(glm::abs(half_extent));
            const float radius = std::max(1e-6f, max_extent);  // half of the longest side
            const float uniform_scale = 1.0f / radius;                // longest side becomes ~2 units

            // Safe view center (fallback to origin)
            glm::vec3 view_center = view_params.center;
            if (!glm::all(glm::isfinite(view_center))) view_center = glm::vec3(0.0f);

            const glm::mat4 T_to_origin = glm::translate(glm::mat4(1.0f), -local.aabb.center());
            const glm::mat4 S_uniform   = glm::scale(glm::mat4(1.0f), glm::vec3(uniform_scale));
            const glm::mat4 T_to_view   = glm::translate(glm::mat4(1.0f), view_center);

            const glm::mat4 model_matrix = T_to_view * S_uniform * T_to_origin;

            Engine::State().emplace_or_replace<TransformComponent>(entity_id, decompose(model_matrix));
            Engine::State().emplace_or_replace<DirtyLocalTransform>(entity_id);

            Log::Info("AABB half_extent: {}, scale: {}", glm::to_string(half_extent), uniform_scale);
        }

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
