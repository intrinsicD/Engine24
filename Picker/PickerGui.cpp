//
// Created by alex on 16.07.24.
//

#include "PickerGui.h"
#include "imgui.h"

namespace Bcg {
    namespace Gui {
        void Show(const Picked &picked) {
            if (ImGui::CollapsingHeader("Entity")) {
                Show(picked.entity);
            }
            ImGui::Text("World Space Point: (%f, %f, %f)", picked.world_space_point.x(), picked.world_space_point.y(),
                        picked.world_space_point.z());
            ImGui::Text("View Space Point: (%f, %f, %f)", picked.view_space_point.x(), picked.view_space_point.y(),
                        picked.view_space_point.z());
            ImGui::Text("NDC Space Point: (%f, %f, %f)", picked.ndc_space_point.x(), picked.ndc_space_point.y(),
                        picked.ndc_space_point.z());
            ImGui::Text("Screen Space Point: (%f, %f)", picked.screen_space_point.x(), picked.screen_space_point.y());
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