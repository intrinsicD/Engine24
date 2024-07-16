//
// Created by alex on 16.07.24.
//

#include "CameraGui.h"
#include "imgui.h"

namespace Bcg::Gui{
    void Show(Camera &camera) {
        if (ImGui::InputFloat3("v_params.center", camera.v_params.center.data())) {
            camera.view = look_at_matrix(camera.v_params.eye, camera.v_params.center, camera.v_params.up);
            camera.v_params.dirty = true;
        }
        if (ImGui::InputFloat3("v_params.eye", camera.v_params.eye.data())) {
            camera.view = look_at_matrix(camera.v_params.eye, camera.v_params.center, camera.v_params.up);
            camera.v_params.dirty = true;
        }
        if (ImGui::InputFloat3("v_params.up", camera.v_params.up.data())) {
            camera.view = look_at_matrix(camera.v_params.eye, camera.v_params.center, camera.v_params.up);
            camera.v_params.dirty = true;
        }

        static int projection_type = 0;

        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            bool changed = ImGui::InputFloat("p_params.fovy", &camera.p_params.fovy);
            changed |= ImGui::InputFloat("p_params.aspect", &camera.p_params.aspect);
            changed |= ImGui::InputFloat("p_params.zNear", &camera.p_params.zNear);
            changed |= ImGui::InputFloat("p_params.zFar", &camera.p_params.zFar);
            if (changed) {
                camera.p_params.dirty = true;
            }
        } else {
            bool changed = ImGui::InputFloat("o_params.left", &camera.o_params.left);
            changed |= ImGui::InputFloat("o_params.right", &camera.o_params.right);
            changed |= ImGui::InputFloat("o_params.bottom", &camera.o_params.bottom);
            changed |= ImGui::InputFloat("o_params.top", &camera.o_params.top);
            changed |= ImGui::InputFloat("o_params.zNear", &camera.o_params.zNear);
            changed |= ImGui::InputFloat("o_params.zFar", &camera.o_params.zFar);
            if (changed) {
                camera.o_params.dirty = true;
            }
        }
        if (ImGui::RadioButton("Perspective", &projection_type, 0)) {
            camera.proj = perspective_matrix(camera.p_params.fovy, camera.p_params.aspect,
                                             camera.p_params.zNear,
                                             camera.p_params.zFar);
            camera.p_params.dirty = true;
        }
        if (ImGui::RadioButton("Orthographic", &projection_type, 1)) {
            camera.proj = ortho_matrix(camera.o_params.left, camera.o_params.right, camera.o_params.bottom,
                                       camera.o_params.top, camera.o_params.zNear, camera.o_params.zFar);
            camera.o_params.dirty = true;
        }

        std::stringstream ss;
        ss << camera.view;
        ImGui::Text("ViewMatrix\n%s", ss.str().c_str());
        //reuse the stringstream
        ss.str("");
        ss << camera.proj;
        ImGui::Text("ProjectionMatrix\n%s", ss.str().c_str());
        if (ImGui::Button("Reset Camera")) {
            camera.v_params = ViewParameters();
            camera.p_params = PerspParameters();
            camera.o_params = OrthoParameters();
            camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
            camera.view = look_at_matrix(camera.v_params.eye, camera.v_params.center, camera.v_params.up);
            camera.proj = perspective_matrix(camera.p_params.fovy, camera.p_params.aspect,
                                             camera.p_params.zNear,
                                             camera.p_params.zFar);
            camera.dirty_view = true;
            camera.dirty_proj = true;
        }
    }
}