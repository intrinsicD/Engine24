//
// Created by alex on 16.07.24.
//

#include "CameraGui.h"
#include "imgui.h"

namespace Bcg::Gui{
    void Show(Camera &camera) {
        bool changed_view = false;
        Vector<float, 3> center = camera.v_params.center();
        if (ImGui::InputFloat3("v_params.center", center.data())) {
            changed_view = true;
            camera.v_params.set_center(center);
        }

        Vector<float, 3> eye = camera.v_params.eye();
        if (ImGui::InputFloat3("v_params.eye", eye.data())) {
            changed_view = true;
            camera.v_params.set_eye(eye);
        }

        Vector<float, 3> up = camera.v_params.up();
        if (ImGui::InputFloat3("v_params.up", up.data())) {
            changed_view = true;
            camera.v_params.set_up(up);
        }

        if(changed_view){
            camera.dirty_view = true;
            camera.view = look_at_matrix(camera.v_params.eye(), camera.v_params.center(), camera.v_params.up());
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
            camera.v_params = ViewParameters<float>();
            camera.p_params = PerspParameters();
            camera.o_params = OrthoParameters();
            camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
            camera.view = look_at_matrix(camera.v_params.eye(), camera.v_params.center(), camera.v_params.up());
            camera.proj = perspective_matrix(camera.p_params.fovy, camera.p_params.aspect,
                                             camera.p_params.zNear,
                                             camera.p_params.zFar);
            camera.dirty_view = true;
            camera.dirty_proj = true;
        }
    }
}