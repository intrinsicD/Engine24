//
// Created by alex on 30.10.24.
//

#include "TransformGui.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "PluginGraphics.h"
#include "Engine.h"
#include "Camera.h"

namespace Bcg::Gui {
    bool Show(Transform<float> &transform) {
        bool changed = false;
        Show(transform.get_params());
        if (ImGui::CollapsingHeader("Matrix")) {
            Show(transform.matrix());
        }
        return changed;
    }

    bool Show(Transform<float>::Parameters &t_params) {
        ImGui::PushID(&t_params);
        ImGui::Text("Scale");
        bool changed = ImGui::InputFloat3("##Scale", t_params.scale.data());
        ImGui::Text("Rotation");
        changed |= ImGui::InputFloat("Angle", &t_params.angle);
        changed |= ImGui::InputFloat3("Axis", t_params.axis.data());
        ImGui::Text("Translation");
        changed |= ImGui::InputFloat3("##Translation", t_params.position.data());
        ImGui::PopID();
        return changed;
    }

    void Show(const Eigen::Matrix<float, 4, 4> &m) {
        std::stringstream ss;
        ss << m;
        ImGui::Text("%s", ss.str().c_str());
    }

    bool Equals(const Eigen::Matrix<float, 4, 4> &m1, const Eigen::Matrix<float, 4, 4> &m2) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (m1(i, j) != m2(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool ShowGuizmo(Eigen::Matrix<float, 4, 4> &mat, Eigen::Matrix<float, 4, 4> &delta, bool &is_scaling) {
        static ImGuizmo::MODE currentGizmoMode(ImGuizmo::LOCAL);
        static ImGuizmo::OPERATION currentGizmoOperation(ImGuizmo::ROTATE);

        if (ImGui::RadioButton("Translate", currentGizmoOperation == ImGuizmo::TRANSLATE))
            currentGizmoOperation = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", currentGizmoOperation == ImGuizmo::ROTATE))
            currentGizmoOperation = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", currentGizmoOperation == ImGuizmo::SCALE))
            currentGizmoOperation = ImGuizmo::SCALE;
        if (ImGui::RadioButton("Universal", currentGizmoOperation == ImGuizmo::UNIVERSAL))
            currentGizmoOperation = ImGuizmo::UNIVERSAL;

        ImGui::Separator();

        is_scaling = true;
        if (currentGizmoOperation != ImGuizmo::SCALE) {
            if (ImGui::RadioButton("Local", currentGizmoMode == ImGuizmo::LOCAL)) {
                currentGizmoMode = ImGuizmo::LOCAL;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("World", currentGizmoMode == ImGuizmo::WORLD)) {
                currentGizmoMode = ImGuizmo::WORLD;
            }
            is_scaling = false;
        }

        static bool use_snap = false;
        static float snap[3] = {1.f, 1.f, 1.f};
        ImGui::Checkbox("use_snap", &use_snap);
        ImGui::SameLine();

        switch (currentGizmoOperation) {
            case ImGuizmo::TRANSLATE:
                ImGui::InputFloat3("Snap", &snap[0]);
                break;
            case ImGuizmo::ROTATE:
                ImGui::InputFloat("Angle Snap", &snap[0]);
                break;
            case ImGuizmo::SCALE:
                ImGui::InputFloat("Scale Snap", &snap[0]);
                break;
        }

        static bool use_bound_sizing = false;
        static bool use_bound_sizing_snap = false;
        static float bounds[] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        static float bounds_snap[] = {0.1f, 0.1f, 0.1f};
        ImGui::Checkbox("use_bound_sizing", &use_bound_sizing);
        ImGui::Checkbox("use_bound_sizing_snap", &use_bound_sizing_snap);

        auto &io = ImGui::GetIO();
        ImGuizmo::SetOrthographic(false);
        auto win_pos = PluginGraphics::get_window_pos();
        auto win_size = PluginGraphics::get_window_size();
        ImGuizmo::SetRect(win_pos.x(), win_pos.y(), io.DisplaySize.x, io.DisplaySize.y);
        auto &camera = Engine::Context().get<Camera<float>>();
        Eigen::Matrix<float, 4, 4> m = mat;
        ImGuizmo::Manipulate(camera.get_view().data(),camera.get_proj().data(), currentGizmoOperation,
                             currentGizmoMode,
                             m.data(), nullptr,
                             use_snap ? &snap[0] : nullptr, use_bound_sizing ? bounds : nullptr,
                             use_bound_sizing_snap ? bounds_snap : nullptr);
        delta = mat.inverse() * m;

        return Equals(delta, Eigen::Matrix<float, 4, 4>::Identity());
    }
}
