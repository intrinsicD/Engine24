//
// Created by alex on 16.07.24.
//

#include "RigidTransformGui.h"
#include "Engine.h"
#include "Camera.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"
#include "AABB.h"
#include "Graphics.h"

namespace Bcg::Gui {
    void Show(RigidTransform &transform) {
        float matrixTranslation[3], matrixRotation[3], matrixScale[3];
        ImGuizmo::DecomposeMatrixToComponents(transform.matrix().data(),
                                              matrixTranslation,
                                              matrixRotation,
                                              matrixScale);
        ImGui::Text("Tr: %f %f %f", matrixTranslation[0], matrixTranslation[1], matrixTranslation[2]);
        ImGui::Text("Rt: %f %f %f", matrixRotation[0], matrixRotation[1], matrixRotation[2]);
        ImGui::Text("Sc: %f %f %f", matrixScale[0], matrixScale[1], matrixScale[2]);
        ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation,
                                                matrixRotation,
                                                matrixScale,
                                                transform.matrix().data());
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
        bool dirty = transform.dirty;
        ImGui::Checkbox("dirty", &dirty);
        bool changed = false;
        if (ImGui::CollapsingHeader("Local")) {
            changed = ShowLocal(transform.local);
        }
        if (ImGui::CollapsingHeader("CachedWorld")) {
            ShowWorld(transform.world);
        }
        return changed;
    }

    bool ShowGuizmo(RigidTransform &transform) {
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

        if (currentGizmoOperation != ImGuizmo::SCALE) {
            if (ImGui::RadioButton("Local", currentGizmoMode == ImGuizmo::LOCAL))
                currentGizmoMode = ImGuizmo::LOCAL;
            ImGui::SameLine();
            if (ImGui::RadioButton("World", currentGizmoMode == ImGuizmo::WORLD))
                currentGizmoMode = ImGuizmo::WORLD;
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
        auto win_pos = Graphics::get_window_pos();
        auto win_size = Graphics::get_window_size();
        ImGuizmo::SetRect(win_pos.x(), win_pos.y(), io.DisplaySize.x, io.DisplaySize.y);
        auto &camera = Engine::Context().get<Camera>();
        return ImGuizmo::Manipulate(camera.view.data(), camera.proj.data(), currentGizmoOperation, currentGizmoMode,
                                    transform.data(), nullptr,
                                    use_snap ? &snap[0] : nullptr, use_bound_sizing ? bounds : nullptr,
                                    use_bound_sizing_snap ? bounds_snap : nullptr);
    }

    bool ShowLocal(RigidTransform &local) {
        bool changed = Edit(local);
        static bool guizmo = false;
        ImGui::Checkbox("Guizmo", &guizmo);
        if (guizmo) {
            changed |= ShowGuizmo(local);
        }
        return changed;
    }

    void ShowWorld(RigidTransform &world) {
        Show(world);
    }
}