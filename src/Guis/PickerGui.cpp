//
// Created by alex on 16.07.24.
//

#include "PickerGui.h"
#include "imgui.h"
#include "GetPrimitives.h"

namespace Bcg {
    namespace Gui {
        void Show(Picked &picked) {
            if (ImGui::CollapsingHeader("Entity")) {
                Show(picked.entity);
            }
        }

        void Show(Picked::Entity &entity) {
            if (ImGui::Checkbox("Show", &entity.show)) {
                auto *vertices = GetPrimitives(entity.id).vertices();
                if (entity.show) {

                } else {

                }
            }
            ImGui::Text("entity id: %u", static_cast<entt::id_type>(entity.id));
            ImGui::Text("is_background: %s", entity.is_background ? "true" : "false");
            ImGui::Text("vertex_idx: %u", entity.vertex_idx);
            ImGui::Text("edge_idx: %u", entity.edge_idx);
            ImGui::Text("face_idx: %u", entity.face_idx);
            ImGui::InputFloat("pick_radius", &entity.pick_radius);
        }
    }
}