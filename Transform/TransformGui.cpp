//
// Created by alex on 16.07.24.
//

#include "TransformGui.h"
#include "Engine.h"
#include "../Camera/Camera.h"
#include "imgui.h"
#include "ImGuizmo.h"

namespace Bcg::Gui{
    static bool enable_guizmo = false;

    void ShowTransform(entt::entity entity_id, bool &show_gui) {
        if (ImGui::Begin("Transform", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
            if (Engine::valid(entity_id) && Engine::has<Transform>(entity_id)) {
                ImGui::Checkbox("Guizmo", &enable_guizmo);
                if (enable_guizmo) {

                    auto &transform = Engine::State().get<Transform>(entity_id);
                    auto &camera = Engine::Context().get<Camera>();
                    auto &view = camera.view;
                    auto &projection = camera.proj;

                    // Define operation and mode (translate, rotate, scale)
                    static ImGuizmo::OPERATION currentOperation(ImGuizmo::TRANSLATE);
                    static ImGuizmo::MODE currentMode(ImGuizmo::WORLD);

                    // Show the Gizmo and handle interactions
                    ImGuiIO &io = ImGui::GetIO();
                    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
                    ImGuizmo::Manipulate(view.data(), projection.data(), currentOperation, currentMode,
                                         transform.data());
                }
                Show(Engine::State().get<Transform>(entity_id));
            }
        }
        ImGui::End();
    }

    void Show(Transform &transform) {
        TransformParameters params = transform.Decompose();
        ImGui::InputFloat3("position", params.position.data());
        ImGui::InputFloat3("rotation", params.angle_axis.data());
        ImGui::InputFloat3("scale", params.scale.data());
        transform = Transform(params);
    }
}