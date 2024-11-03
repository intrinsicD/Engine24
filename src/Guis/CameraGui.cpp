//
// Created by alex on 16.07.24.
//

#include "CameraGui.h"
#include "imgui.h"
#include "PluginGraphics.h"

namespace Bcg::Gui {
    bool Show(ViewParams &v_params) {
        bool changed_v = ImGui::InputFloat3("v_params.center", glm::value_ptr(v_params.center));
        changed_v |= ImGui::InputFloat3("v_params.eye", glm::value_ptr(v_params.eye));
        changed_v |= ImGui::InputFloat3("v_params.up", glm::value_ptr(v_params.up));
        return changed_v;
    }

    bool Show(PerspectiveParams &p_params) {
        bool changed_p = ImGui::InputFloat("p_params.fovy", &p_params.fovy_degrees);
        changed_p |= ImGui::InputFloat("p_params.aspect", &p_params.aspect);
        changed_p |= ImGui::InputFloat("p_params.zNear", &p_params.zNear);
        changed_p |= ImGui::InputFloat("p_params.zFar", &p_params.zFar);
        return changed_p;
    }

    bool Show(OrthoParams &o_params) {
        bool changed_o = ImGui::InputFloat("o_params.left", &o_params.left);
        changed_o |= ImGui::InputFloat("o_params.right", &o_params.right);
        changed_o |= ImGui::InputFloat("o_params.bottom", &o_params.bottom);
        changed_o |= ImGui::InputFloat("o_params.top", &o_params.top);
        changed_o |= ImGui::InputFloat("o_params.zNear", &o_params.zNear);
        changed_o |= ImGui::InputFloat("o_params.zFar", &o_params.zFar);
        return changed_o;
    }

    void ShowMatrix(const glm::mat4 &m) {
        ImGui::Text("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f",
                    m[0][0], m[1][0], m[2][0], m[3][0],
                    m[0][1], m[1][1], m[2][1], m[3][1],
                    m[0][2], m[1][2], m[2][2], m[3][2],
                    m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    void ShowProjection(Camera &camera) {
        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            PerspectiveParams p_params = GetPerspectiveParams(camera);
            if (Show(p_params)) {
                SetPerspectiveParams(camera, p_params);
            }
            static float depth = p_params.zNear;
            bool changed_depth = ImGui::SliderFloat("Depth", &depth, p_params.zNear, p_params.zFar);
            if (ImGui::Button("Convert to Ortho") || changed_depth) {
                {
                    OrthoParams o_params = Convert(p_params, depth);
                    SetOrthoParams(camera, o_params);
                }
            } else {
                OrthoParams o_params = GetOrthoParams(camera);
                if (Show(o_params)) {
                    SetOrthoParams(camera, o_params);
                }
                if (ImGui::Button("Convert to Perspective")) {
                    PerspectiveParams p_params = Convert(o_params);
                    SetPerspectiveParams(camera, p_params);
                }
            }
        }
    }

    void Show(Camera &camera) {
        ViewParams view_params = GetViewParams(camera);
        if (Show(view_params)) {
            SetViewParams(camera, view_params);
        }
        if (ImGui::CollapsingHeader("ViewMatrix")) {
            ShowMatrix(camera.view);
        }

        ShowProjection(camera);

        if (ImGui::CollapsingHeader(
                camera.proj_type == Camera::ProjectionType::PERSPECTIVE ? "Perspective ProjectionMatrix"
                                                                        : "Ortho ProjectionMatrix")) {
            ShowMatrix(camera.proj);
        }

        if (ImGui::Button("Reset Camera")) {
            view_params.eye = glm::vec3(0, 0, 1);
            view_params.center = glm::vec3(0, 0, 0);
            view_params.up = glm::vec3(0, 1, 0);
            camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
            SetViewParams(camera, view_params);

            auto vp = PluginGraphics::get_viewport();
            auto viewport_width = vp[2];
            auto viewport_height = vp[3];

            float aspect_ratio = float(viewport_width) / float(viewport_height);
            PerspectiveParams p_params = GetPerspectiveParams(camera);
            p_params.fovy_degrees = 45.0f;
            p_params.aspect = aspect_ratio;
            p_params.zNear = 0.1f;
            p_params.zFar = 100.0f;
            SetPerspectiveParams(camera, p_params);
        }
    }
}