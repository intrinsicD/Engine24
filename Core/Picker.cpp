//
// Created by alex on 03.07.24.
//

#include "Picker.h"
#include "Engine.h"
#include "imgui.h"

namespace Bcg {
    Picker::Picker() : Plugin("Picker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
    }

    Picked &Picker::pick(double x, double y) {
        return last_picked();
    }

    Picked &Picker::last_picked() {
        return Engine::Context().get<Picked &>();
    }

    namespace Gui {
        void Show(const Picked &picked) {
            Show(picked.entity);
        }

        void Show(const Picked::Entity &entity) {
            ImGui::Text("entity id: %u", static_cast<entt::id_type>(entity.id));
            ImGui::Text("is_background: %s", entity.is_background ? "true" : "false");
            ImGui::Text("vertex_idx: %u", entity.vertex_idx);
            ImGui::Text("edge_idx: %u", entity.edge_idx);
            ImGui::Text("face_idx: %u", entity.face_idx);
        }
    }


}