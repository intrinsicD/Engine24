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
#include "PluginViewSphere.h"
#include "PropertyEigenMap.h"

namespace Bcg {
    static void on_picked_vertex(const Events::PickedVertex &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (keyboard.strg() && keyboard.shift()) {
            auto &selection = Engine::require<Selection>(event.entity_id);
            for (auto idx: *event.idx) {
                auto iter = selection.vertices.find(idx);
                if (iter != selection.vertices.end()) {
                    selection.vertices.erase(iter);
                } else {
                    selection.vertices.emplace(idx);
                }
            }
            Commands::MarkPoints(event.entity_id, "v::selected").execute();
        }
    }

    static void on_picked_background(const Events::PickedBackgound &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (!keyboard.strg() && keyboard.shift()) {
            auto &picked = Engine::Context().get<Picked>();
            auto entity_id = picked.entity.id;
            auto &selection = Engine::require<Selection>(entity_id);
            selection.vertices.clear();
            Commands::MarkPoints(entity_id, "v::selected").execute();
        }
    }

    PluginSelection::PluginSelection() : Plugin("Selection") {}

    void PluginSelection::activate() {
        if (base_activate()) {
            Engine::Dispatcher().sink<Events::PickedVertex>().connect<&on_picked_vertex>();
            Engine::Dispatcher().sink<Events::PickedBackgound>().connect<&on_picked_background>();
        }
    }

    void PluginSelection::begin_frame() {}

    void PluginSelection::update() {}

    void PluginSelection::end_frame() {}

    void PluginSelection::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::PickedVertex>().disconnect<&on_picked_vertex>();
            Engine::Dispatcher().sink<Events::PickedBackgound>().disconnect<&on_picked_background>();
        }
    }

    bool show_selection_gui = false;

    void PluginSelection::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("Selection", nullptr, &show_selection_gui);
            ImGui::EndMenu();
        }
    }

    void PluginSelection::render_gui() {
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

    namespace Commands {
        void MarkPoints::execute() const {
            if (!Engine::has<Selection>(entity_id)) {
                return;
            }
            auto *vertices = GetPrimitives(entity_id).vertices();
            auto selected_vertices = vertices->get_or_add<Vector<float, 3>>(property_name, Vector<float, 3>(1.0f));

            Map(selected_vertices.vector()).setOnes();
            auto &selection = Engine::State().get<Selection>(entity_id);
            for (auto idx: selection.vertices) {
                selected_vertices[idx] = {1.0, 0.0, 0.0};
            }

            SetColorSphereView(entity_id, property_name).execute();
        }

        void EnableVertexSelection::execute() const {
            Log::TODO("Implement EnableVertexSelection.");
        }
    }
}