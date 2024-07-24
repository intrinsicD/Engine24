//
// Created by alex on 16.07.24.
//

#include "TransformGui.h"
#include "Engine.h"
#include "../Camera/Camera.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"
#include "AABB.h"
#include "Graphics.h"

namespace Bcg::Gui {
    static bool enable_guizmo = false;

    void EditTransform(float *cameraView, float *cameraProjection, float *matrix, float *delta = nullptr) {
        static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::LOCAL);
        static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::ROTATE);
        static bool useSnap = false;
        static float snap[3] = {1.f, 1.f, 1.f};
        static float bounds[] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        static float boundsSnap[] = {0.1f, 0.1f, 0.1f};
        static bool boundSizing = false;
        static bool boundSizingSnap = false;

        static bool editTransformDecomposition = false;
        ImGui::Checkbox("edit decomposition", &editTransformDecomposition);

        if (editTransformDecomposition) {
            if (ImGui::IsKeyPressed(ImGuiKey_T))
                mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_E))
                mCurrentGizmoOperation = ImGuizmo::ROTATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R)) // r Key
                mCurrentGizmoOperation = ImGuizmo::SCALE;
            if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
                mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
                mCurrentGizmoOperation = ImGuizmo::ROTATE;
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
                mCurrentGizmoOperation = ImGuizmo::SCALE;
            if (ImGui::RadioButton("Universal", mCurrentGizmoOperation == ImGuizmo::UNIVERSAL))
                mCurrentGizmoOperation = ImGuizmo::UNIVERSAL;

            float matrixTranslation[3], matrixRotation[3], matrixScale[3];
            ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
            ImGui::InputFloat3("Tr", matrixTranslation);
            ImGui::InputFloat3("Rt", matrixRotation);
            ImGui::InputFloat3("Sc", matrixScale);
            ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

            if (mCurrentGizmoOperation != ImGuizmo::SCALE) {
                if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
                    mCurrentGizmoMode = ImGuizmo::LOCAL;
                ImGui::SameLine();
                if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
                    mCurrentGizmoMode = ImGuizmo::WORLD;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_S))
                useSnap = !useSnap;
            ImGui::Checkbox("##UseSnap", &useSnap);
            ImGui::SameLine();

            switch (mCurrentGizmoOperation) {
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
            ImGui::Checkbox("Bound Sizing", &boundSizing);
            if (boundSizing) {
                ImGui::PushID(3);
                ImGui::Checkbox("##BoundSizing", &boundSizingSnap);
                ImGui::SameLine();
                ImGui::InputFloat3("Snap", boundsSnap);
                ImGui::PopID();
            }
        }

        ImGuiIO &io = ImGui::GetIO();
        auto size = Graphics::get_window_size();
        io.DisplaySize.x = size[0];
        io.DisplaySize.y = size[1];
        static ImGuiWindowFlags gizmoWindowFlags = 0;
        ImGuizmo::SetOrthographic(false);
        static bool useWindow = false;
        ImGui::Checkbox("Use Window", &useWindow);
        if (useWindow) {
            ImGui::SetNextWindowSize(ImVec2(800, 400), ImGuiCond_Appearing);
            ImGui::SetNextWindowPos(ImVec2(400, 20), ImGuiCond_Appearing);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, (ImVec4) ImColor(0.35f, 0.3f, 0.3f));
            ImGui::Begin("Gizmo", 0, gizmoWindowFlags);
            ImGuizmo::SetDrawlist();
            float windowWidth = (float) ImGui::GetWindowWidth();
            float windowHeight = (float) ImGui::GetWindowHeight();
            ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);
            ImGuiWindow *window = ImGui::GetCurrentWindow();
            gizmoWindowFlags =
                    ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(window->InnerRect.Min, window->InnerRect.Max)
                    ? ImGuiWindowFlags_NoMove : 0;
        } else {
            ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
        }

        static bool show_grid = false;
        ImGui::Checkbox("show_grid", &show_grid);

        if (show_grid) {
            Matrix<float, 4, 4> id = Matrix<float, 4, 4>::Identity();
            static float grid_size = 100.f;
            ImGui::InputFloat("grid_size", &grid_size);
            ImGuizmo::DrawGrid(cameraView, cameraProjection, id.data(), grid_size);
        }

        static bool show_cube = false;
        ImGui::Checkbox("show_cube", &show_cube);
        if (show_cube) {
            ImGuizmo::DrawCubes(cameraView, cameraProjection, matrix, 1);
        }

        ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix, delta,
                             useSnap ? &snap[0] : NULL, boundSizing ? bounds : NULL,
                             boundSizingSnap ? boundsSnap : NULL);

        if (useWindow) {
            ImGui::End();
            ImGui::PopStyleColor(1);
        }
    }

    void ShowTransform(entt::entity entity_id, bool &show_gui) {
        if (ImGui::Begin("Transform", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
            if (Engine::valid(entity_id) && Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                auto &aabb = Engine::State().get<AABB>(entity_id);
                auto &camera = Engine::Context().get<Camera>();
                auto &view = camera.view;
                auto &projection = camera.proj;
                Matrix<float, 4, 4> delta = Matrix<float, 4, 4>::Identity();
                Matrix<float, 4, 4> matrix = Transform::Translation(aabb.center()).matrix();
                EditTransform(view.data(), projection.data(), matrix.data(), delta.data());
                transform.matrix() = transform * delta;
            }
        }
        ImGui::End();
    }

    void Show(Transform &transform) {
/*        TransformParameters params = transform.Decompose();
        ImGui::InputFloat3("position", params.position.data());
        ImGui::InputFloat3("rotation", params.angle_axis.data());
        ImGui::InputFloat3("scale", params.scale.data());
        transform = Transform(params);*/
    }
}