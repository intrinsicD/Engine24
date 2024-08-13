//
// Created by alex on 06.08.24.
//

#include "SelectionGui.h"
#include "Engine.h"
#include "PluginSphereView.h"
#include "imgui.h"

namespace Bcg::Gui {
    void ShowSelection(entt::entity entity_id) {
        if (Engine::valid(entity_id) && Engine::has<Selection>(entity_id)) {
            auto &selection = Engine::require<Selection>(entity_id);

            Show(selection);
        }
    }

    void Show(Selection &selection) {
        static std::pair<int, std::string> curr = {0, "empty"};

        std::string labels;
        if (selection.vertices.empty()) {
           labels = "empty";
        } else {
            for (auto idx: selection.vertices) {
               labels += std::to_string(idx) + " ";
            }

        }
        ImGui::Text("Selected Vertices: %s", labels.c_str());
    }
}