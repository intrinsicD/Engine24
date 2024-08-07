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
#include "SelectionCommands.h"

namespace Bcg {
    static void on_picked_vertex(const Events::PickedVertex &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (keyboard.strg() && keyboard.shift()) {
            auto &selection = Engine::require<Selection>(event.entity_id);
            auto iter = selection.vertices.find(event.idx);
            if (iter != selection.vertices.end()) {
                selection.vertices.erase(iter);
            } else {
                selection.vertices.emplace(event.idx);
            }
            Commands::MarkPoints(event.entity_id, "v::selected").execute();
        }
    }

    static void on_picked_background(const Events::PickedBackgound &event){
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
        Plugin::activate();
        Engine::Dispatcher().sink<Events::PickedVertex>().connect<&on_picked_vertex>();
        Engine::Dispatcher().sink<Events::PickedBackgound>().connect<&on_picked_background>();
    }

    void PluginSelection::begin_frame() {}

    void PluginSelection::update() {}

    void PluginSelection::end_frame() {}

    void PluginSelection::deactivate() {
        Plugin::deactivate();
        Engine::Dispatcher().sink<Events::PickedVertex>().disconnect<&on_picked_vertex>();
        Engine::Dispatcher().sink<Events::PickedBackgound>().disconnect<&on_picked_background>();
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

    void PluginSelection::render() {}
}