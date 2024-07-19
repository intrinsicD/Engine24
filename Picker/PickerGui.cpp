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
            const auto &points = picked.spaces;
            ImGui::Text("ScreenSpacePos: %lf, %lf", points.ssp.x(), points.ssp.y());
            ImGui::Text("ScreenSpacePosDpiAdjusted: %lf, %lf", points.sspda.x(), points.sspda.y());
            ImGui::Text("NdcSpacePos: %lf, %lf, %lf", points.ndc.x(), points.ndc.y(), points.ndc.z());
            ImGui::Text("ViewSpacePos: %lf, %lf, %lf", points.vsp.x(), points.vsp.y(), points.vsp.z());
            ImGui::Text("WorldSpacePos: %lf, %lf, %lf", points.wsp.x(), points.wsp.y(), points.wsp.z());
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