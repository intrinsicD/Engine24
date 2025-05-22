//
// Created by alex on 16.07.24.
//

#include "CameraGui.h"
#include "imgui.h"
#include "PluginGraphics.h"

namespace Bcg::Gui {
    bool Show(Camera<float>::ViewParams &v_params) {
        bool changed_v = ImGui::InputFloat3("v_params.center", v_params.center.data());
        changed_v |= ImGui::InputFloat3("v_params.eye", v_params.eye.data());
        changed_v |= ImGui::InputFloat3("v_params.up", v_params.up.data());
        return changed_v;
    }

    bool Show(Camera<float>::PerspectiveParams &p_params) {
        bool changed_p = ImGui::InputFloat("p_params.fovy", &p_params.fovy_degrees);
        changed_p |= ImGui::InputFloat("p_params.aspect", &p_params.aspect);
        changed_p |= ImGui::InputFloat("p_params.zNear", &p_params.zNear);
        changed_p |= ImGui::InputFloat("p_params.zFar", &p_params.zFar);
        return changed_p;
    }

    bool Show(Camera<float>::OrthoParams &o_params) {
        bool changed_o = ImGui::InputFloat("o_params.left", &o_params.left);
        changed_o |= ImGui::InputFloat("o_params.right", &o_params.right);
        changed_o |= ImGui::InputFloat("o_params.bottom", &o_params.bottom);
        changed_o |= ImGui::InputFloat("o_params.top", &o_params.top);
        changed_o |= ImGui::InputFloat("o_params.zNear", &o_params.zNear);
        changed_o |= ImGui::InputFloat("o_params.zFar", &o_params.zFar);
        return changed_o;
    }

    void ShowMatrix(const Eigen::Matrix<float, 4, 4> &m) {
        std::stringstream ss;
        ss << m;
        ImGui::Text("%s", ss.str().c_str());
    }

    void ShowProjection(Camera<float> &camera) {
        auto projection_type = camera.get_projection_type();
        if (projection_type == Camera<float>::ProjectionType::PERSPECTIVE) {
            auto p_params = camera.get_perspective_params();
            if (Show(p_params)) {
                camera.set_perspective_params(p_params);
            }
        }else{
            auto o_params = camera.get_ortho_params();
            if (Show(o_params)) {
                camera.set_ortho_params(o_params);
            }
        }
    }

    void Show(Camera<float> &camera) {
        auto view_params = camera.get_view_params();
        if (Show(view_params)) {
            camera.set_view_params(view_params);
        }
        if (ImGui::CollapsingHeader("ViewMatrix")) {
            ShowMatrix(camera.get_view());
        }

        ShowProjection(camera);

        auto projection_type = camera.get_projection_type();
        if (ImGui::CollapsingHeader(
                projection_type == Camera<float>::ProjectionType::PERSPECTIVE ? "Perspective ProjectionMatrix"
                                                                        : "Ortho ProjectionMatrix")) {
            ShowMatrix(camera.get_proj());
        }

        if (ImGui::Button("Reset Camera")) {
            auto view_params = camera.get_view_params();
            view_params.eye = Eigen::Vector<float, 3>(0, 0, 1);
            view_params.center = Eigen::Vector<float, 3>(0, 0, 0);
            view_params.up = Eigen::Vector<float, 3>(0, 1, 0);
            camera.set_view_params(view_params);

            auto vp = PluginGraphics::get_viewport();
            auto viewport_width = vp[2];
            auto viewport_height = vp[3];

            float aspect_ratio = float(viewport_width) / float(viewport_height);

            auto perspective_params = camera.get_perspective_params();
            perspective_params.fovy_degrees = 45.0f;
            perspective_params.aspect = aspect_ratio;
            perspective_params.zNear = 0.1f;
            perspective_params.zFar = 100.0f;
            camera.set_perspective_params(perspective_params);
        }
    }
}