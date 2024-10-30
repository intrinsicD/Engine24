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
    bool Show(Transform &transform) {
        bool changed = false;
        ImGui::Text("Local");
        Show(transform.local());
        ImGui::Separator();
        TransformParameters t_params = decompose(transform.local());
        if (Show(t_params)) {
            transform.set_local(compose(t_params));
            changed = true;
        }
        ImGui::Text("World");
        Show(transform.world());
        ImGui::Separator();
        TransformParameters w_params = decompose(transform.world());
        Show(w_params);
        if (ImGui::CollapsingHeader("Cached Parent World")) {
            Show(transform.get_cached_parent_world());
        }
        return changed;
    }

    bool Show(TransformParameters &t_params) {
        ImGui::PushID(&t_params);
        ImGui::Text("Scale");
        bool changed = ImGui::InputFloat3("##Scale", glm::value_ptr(t_params.scale));
        ImGui::Text("Rotation");
        changed |= ImGui::InputFloat3("##Rotation", glm::value_ptr(t_params.angle_axis));
        ImGui::Text("Translation");
        changed |= ImGui::InputFloat3("##Translation", glm::value_ptr(t_params.position));
        ImGui::PopID();
        return changed;
    }

    void Show(const glm::mat4 &m) {
        ImGui::Text("%f %f %f %f\n"
                    "%f %f %f %f\n"
                    "%f %f %f %f\n"
                    "%f %f %f %f",
                    m[0][0], m[1][0], m[2][0], m[3][0],
                    m[0][1], m[1][1], m[2][1], m[3][1],
                    m[0][2], m[1][2], m[2][2], m[3][2],
                    m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    bool Equals(const glm::mat4 &m1, const glm::mat4 &m2) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (m1[i][j] != m2[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool ShowGuizmo(glm::mat4 &mat, glm::mat4 &delta, bool &is_scaling) {
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
        ImGuizmo::SetRect(win_pos.x, win_pos.y, io.DisplaySize.x, io.DisplaySize.y);
        auto &camera = Engine::Context().get<Camera>();
        Matrix<float, 4, 4> m = mat;
        ImGuizmo::Manipulate(glm::value_ptr(camera.view), glm::value_ptr(camera.proj), currentGizmoOperation, currentGizmoMode,
                             glm::value_ptr(m), nullptr,
                             use_snap ? &snap[0] : nullptr, use_bound_sizing ? bounds : nullptr,
                             use_bound_sizing_snap ? bounds_snap : nullptr);
        delta = glm::inverse(mat) * m;

        return Equals(delta, glm::mat4(1.0f));
    }
}