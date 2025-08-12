//
// Created by alex on 06.08.24.
//

#include "PluginSelection.h"
#include "Engine.h"
#include "imgui.h"
#include "Keyboard.h"
#include "SelectionGui.h"
#include "Picker.h"
#include "EventsPicker.h"
#include "GetPrimitives.h"
#include "ModuleSphereView.h"
#include "PropertyEigenMap.h"

namespace Bcg {
    static void insert_or_remove_set(std::set<size_t> &unique_set, const std::vector<size_t> &indices) {
        for (auto idx: indices) {
            auto iter = unique_set.find(idx);
            if (iter != unique_set.end()) {
                unique_set.erase(iter);
            } else {
                unique_set.emplace(idx);
            }
        }
    }

    static void on_picked_vertex(const Events::PickedVertex &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (keyboard.strg() && keyboard.shift()) {
            auto mode = PluginSelection::get_current_selection_mode();
            switch (mode) {
                case SelectionMode::Vertex: {
                    auto &selection = PluginSelection::get_selected_vertices(event.entity_id);
                    if (event.idx->empty()) {
                        selection.vertices.clear();
                    } else {
                        insert_or_remove_set(selection.vertices, *event.idx);
                    }
                    PluginSelection::mark_selected_vertices(event.entity_id, Vector<float, 3>(1.0f, 0.0f, 0.0f));
                    break;
                }
                case SelectionMode::Edge: {
                }
                case SelectionMode::Face: {
                }
                default:{}
            }
        }
    }

    static void on_picked_background(const Events::PickedBackgound &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (!keyboard.strg() && keyboard.shift()) {
            auto &picked = Engine::Context().get<Picked>();
            auto entity_id = picked.entity.id;
            auto &selection = Engine::require<Selection>(entity_id);
            selection.vertices.clear();

            auto mode = PluginSelection::get_current_selection_mode();
            switch (mode) {
                case SelectionMode::Vertex: {
                    auto &selection = PluginSelection::get_selected_vertices(entity_id);
                    selection.vertices.clear();
                    PluginSelection::mark_selected_vertices(entity_id, Vector<float, 3>(1.0f, 0.0f, 0.0f));
                    break;
                }
                case SelectionMode::Edge: {
                    auto &selection = PluginSelection::get_selected_edges(entity_id);
                    selection.edges.clear();
                    PluginSelection::mark_selected_edges(entity_id, Vector<float, 3>(1.0f, 0.0f, 0.0f));
                    break;
                }
                case SelectionMode::Face: {
                    auto &selection = PluginSelection::get_selected_faces(entity_id);
                    selection.faces.clear();
                    PluginSelection::mark_selected_faces(entity_id, Vector<float, 3>(1.0f, 0.0f, 0.0f));
                    break;
                }
            }
        }
    }

    PluginSelection::PluginSelection() : Plugin("Selection") {
    }

    void PluginSelection::activate() {
        if (base_activate()) {
            Engine::Context().emplace<SelectionMode>();
            Engine::Context().emplace<SelectionMode>();
            Engine::Dispatcher().sink<Events::PickedVertex>().connect<&on_picked_vertex>();
            Engine::Dispatcher().sink<Events::PickedBackgound>().connect<&on_picked_background>();
        }
    }

    void PluginSelection::begin_frame() {
    }

    void PluginSelection::update() {
    }

    void PluginSelection::end_frame() {
    }

    void PluginSelection::deactivate() {
        if (base_deactivate()) {
            Engine::Context().erase<SelectionMode>();
            Engine::Context().erase<SelectionMode>();
            Engine::Dispatcher().sink<Events::PickedVertex>().disconnect<&on_picked_vertex>();
            Engine::Dispatcher().sink<Events::PickedBackgound>().disconnect<&on_picked_background>();
        }
    }

    bool show_selection_gui = false;
    bool show_selection_mode_gui = false;

    void PluginSelection::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            if (ImGui::BeginMenu("Selection")) {
                ImGui::MenuItem("Current", nullptr, &show_selection_gui);
                ImGui::MenuItem("Modes", nullptr, &show_selection_mode_gui);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    static const char *ModeLabel(SelectionMode mode) {
        switch (mode) {
            case SelectionMode::None:
                return "None";
            case SelectionMode::Entity:
                return "Entity";
            case SelectionMode::Vertex:
                return "Vertex";
            case SelectionMode::Edge:
                return "Edge";
            case SelectionMode::Face:
                return "Face";
        }
        return "Unknown";
    }

    static bool show_gui(SelectionMode &current_mode) {
        bool mode_changed = false;

        // 1) Compute the position at the lower‐left corner:
        // 1) Get the main viewport (this corresponds to your GLFW window)
        ImGuiViewport *viewport = ImGui::GetMainViewport();
        // WorkPos is the top‐left corner of the drawable area (excludes OS window decorations)
        ImVec2 work_pos = viewport->WorkPos;
        ImVec2 work_size = viewport->WorkSize;

        // 2) Compute the position at the lower‐left corner of that viewport:
        // We want the window’s top‐left to sit at (work_pos.x, work_pos.y + work_size.y)
        ImVec2 target_pos = ImVec2(work_pos.x, work_pos.y + work_size.y);

        // Use pivot (0, 1) so that ImGui treats target_pos as the window’s top‐left corner,
        // which ends up at the bottom-left of the viewport.
        ImGui::SetNextWindowPos(target_pos, ImGuiCond_Always, ImVec2(0.0f, 1.0f));

        // 2) Window flags: no title, no resize, no move, no navigation focus, no saved settings
        constexpr ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoBackground |
                ImGuiWindowFlags_NoSavedSettings;

        // 3) Begin a tiny window; we hide the title by using "##" prefix
        ImGui::Begin("##SelectionModeRadio", nullptr, flags);

        // We want no padding around the children, so they align flush with the window border:
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));

        // 4) Render four radio buttons, one per mode:
        //    We store the current mode as an int (0…3) for ImGui::RadioButton
        int mode_int = static_cast<int>(current_mode);
        ImGui::Text("Selection Mode:");
        for (int i = 0; i < 5; ++i) {
            SelectionMode mode = static_cast<SelectionMode>(i);
            const char *label = ModeLabel(mode);

            // ImGui::RadioButton returns true if clicked
            if (ImGui::RadioButton(label, mode_int == i)) {
                if (mode_int != i) {
                    mode_int = i;
                    mode_changed = true;
                }
            }
            ImGui::SameLine();
            // Put each radio button on its own line (default behavior),
            // so nothing special is needed here.
        }

        // 5) Apply the new mode back to the enum
        if (mode_changed) {
            current_mode = static_cast<SelectionMode>(mode_int);
        }

        ImGui::PopStyleVar(); // pop WindowPadding
        ImGui::End();
        return mode_changed;
    }

    void PluginSelection::render_gui() {
        if (show_selection_mode_gui) {
            auto &current_mode = Engine::Context().get<SelectionMode>();
            show_gui(current_mode);
        }
        if (show_selection_gui) {
            if (ImGui::Begin("Selection", &show_selection_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                auto entity_id = picked.entity.id;
                Gui::ShowSelection(entity_id);
            }
            ImGui::End();
        }
    }

    void PluginSelection::render() {
    }

    SelectionMode PluginSelection::get_current_selection_mode() {
        return Engine::Context().get<SelectionMode>();
    }

    void PluginSelection::set_current_selection_mode(SelectionMode mode) {
        Engine::Context().get<SelectionMode>() = mode;
    }


    void PluginSelection::disable_selection() {
        Engine::Context().get<SelectionMode>() = SelectionMode::None;
    }

    SelectedEntities &PluginSelection::get_selected_entities() {
        return Engine::Context().get<SelectedEntities>();
    }

    void PluginSelection::mark_selected_vertices(entt::entity entity_id, const Vector<float, 3> &color) {
        if (!Engine::valid(entity_id)) return;
        auto &selection = get_selected_vertices(entity_id);
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;
        auto selected_vertices = vertices->get_or_add<Vector<float, 3> >("v::selected", Vector<float, 3>(1.0f));
        Map(selected_vertices.vector()).setOnes();
        for (auto idx: selection.vertices) {
            if (idx < selected_vertices.vector().size()) {
                selected_vertices[idx] = color;
            }
        }
        ModuleSphereView::set_color(entity_id, "v::selected");
    }


    SelectedVertices &PluginSelection::get_selected_vertices(entt::entity entity_id) {
        return Engine::State().get_or_emplace<SelectedVertices>(entity_id);
    }

    void PluginSelection::mark_selected_edges(entt::entity entity_id, const Vector<float, 3> &color) {
        if (!Engine::valid(entity_id)) return;
        auto &selection = get_selected_edges(entity_id);
        auto *edges = GetPrimitives(entity_id).edges();
        if (!edges) return;
        auto selected_edges = edges->get_or_add<Vector<float, 3> >("e::selected", Vector<float, 3>(1.0f));
        Map(selected_edges.vector()).setOnes();
        for (auto idx: selection.edges) {
            if (idx < selected_edges.vector().size()) {
                selected_edges[idx] = color;
            }
        }
        //TODO: Implement edge color setting in ModuleGraphView (Which is not implemented yet)
    }

    SelectedEdges &PluginSelection::get_selected_edges(entt::entity entity_id) {
        return Engine::State().get_or_emplace<SelectedEdges>(entity_id);
    }

    void PluginSelection::mark_selected_faces(entt::entity entity_id, const Vector<float, 3> &color) {
        if (!Engine::valid(entity_id)) return;
        auto &selection = get_selected_faces(entity_id);
        auto *faces = GetPrimitives(entity_id).faces();
        if (!faces) return;
        auto selected_faces = faces->get_or_add<Vector<float, 3> >("f::selected", Vector<float, 3>(1.0f));
        Map(selected_faces.vector()).setOnes();
        for (auto idx: selection.faces) {
            if (idx < selected_faces.vector().size()) {
                selected_faces[idx] = color;
            }
        }
        //TODO: Implement edge color setting in ModuleMeshView (Which is implemented, but face based color is not yet supported)
    }

    SelectedFaces &PluginSelection::get_selected_faces(entt::entity entity_id) {
        return Engine::State().get_or_emplace<SelectedFaces>(entity_id);
    }

    namespace Commands {
        void MarkPoints::execute() const {
            PluginSelection::mark_selected_vertices(entity_id, Vector<float, 3>(1.0f, 0.0f, 0.0f));
        }

        void EnableVertexSelection::execute() const {
            Log::TODO("Implement EnableVertexSelection.");
        }
    }
}
