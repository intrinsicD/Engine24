//
// Created by alex on 16.07.24.
//

#include "RigidTransformGui.h"
#include "Engine.h"
#include "Camera.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"
#include "PluginGraphics.h"

namespace Bcg::Gui {
    void Show(const RigidTransform &transform) {
        float matrixTranslation[3], matrixRotation[3], matrixScale[3];
        ImGuizmo::DecomposeMatrixToComponents(transform.matrix().data(),
                                              matrixTranslation,
                                              matrixRotation,
                                              matrixScale);
        ImGui::Text("Tr: %f %f %f", matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);
        ImGui::Text("Rt: %f %f %f", matrixRotation[0], matrixRotation[1], matrixRotation[2]);
        ImGui::Text("Sc: %f %f %f", matrixScale[0], matrixScale[1], matrixScale[2]);
    }

    bool Edit(RigidTransform &transform) {
        float matrixTranslation[3], matrixRotation[3], matrixScale[3];
        ImGuizmo::DecomposeMatrixToComponents(transform.matrix().data(), matrixTranslation, matrixRotation,
                                              matrixScale);
        bool changed = false;
        changed |= ImGui::InputFloat3("Tr", matrixTranslation);
        changed |= ImGui::InputFloat3("Rt", matrixRotation);
        changed |= ImGui::InputFloat3("Sc", matrixScale);
        ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale,
                                                transform.matrix().data());
        return changed;
    }

    bool Show(Transform &transform) {
        bool dirty = transform.is_dirty();
        ImGui::Checkbox("dirty", &dirty);
        bool changed = false;
        static bool guizmo = false;
        ImGui::Checkbox("Guizmo", &guizmo);
        if (guizmo) {
            RigidTransform delta = RigidTransform::Identity();
            bool is_scaling = false;
            if (ShowGuizmo(transform.world(), delta, is_scaling)) {
                //Todo think how to handle scaling a parent and what to do with a child...
                transform.set_local(transform.local.matrix() * delta.matrix());

                changed = true;
            }
        }
        if (ImGui::CollapsingHeader("Local")) {
            if (ImGui::Button("Reset Identity")) {
                transform.set_local_identity();
                changed = true;
            }
            if (ShowLocal(transform.local)) {
                transform.set_local(transform.local.matrix());
                changed = true;
            }
        }
        if (ImGui::CollapsingHeader("CachedWorld")) {
            ShowWorld(transform.world());
        }
        return changed;
    }

    bool ShowGuizmo(const RigidTransform &transform, RigidTransform &delta, bool &is_scaling) {
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
        auto &camera = Engine::Context().get<Camera>();
        Matrix<float, 4, 4> mat = transform.matrix();
        ImGuizmo::Manipulate(camera.view.data(), camera.proj.data(), currentGizmoOperation, currentGizmoMode,
                             mat.data(), nullptr,
                             use_snap ? &snap[0] : nullptr, use_bound_sizing ? bounds : nullptr,
                             use_bound_sizing_snap ? bounds_snap : nullptr);
        delta = transform.inverse() * mat;
        return delta.matrix() != Matrix<float, 4, 4>::Identity();
    }

    bool ShowLocal(RigidTransform &transform) {
        return Edit(transform);
    }

    void ShowWorld(const RigidTransform &transform) {
        Show(transform);
    }
}