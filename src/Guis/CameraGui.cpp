//
// Created by alex on 16.07.24.
//

#include "CameraGui.h"
#include "imgui.h"

namespace Bcg::Gui{
    void Show(Camera &camera) {
        bool changed_v = false;
        ViewParams view_params = get_view_params(camera);
        if (ImGui::InputFloat3("v_params.center", glm::value_ptr(view_params.center))) {
            changed_v = true;
        }
        if (ImGui::InputFloat3("v_params.eye", glm::value_ptr(view_params.eye))) {
            changed_v = true;
        }
        if (ImGui::InputFloat3("v_params.up", glm::value_ptr(view_params.up))) {
            changed_v = true;
        }

        if(changed_v){
            set_view_params(camera, view_params);
        }

        static int projection_type = 0;

        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            PerspectiveParams p_params = get_perspective_params(camera);
            bool changed_p = ImGui::InputFloat("p_params.fovy", &p_params.fovy);
            changed_p |= ImGui::InputFloat("p_params.aspect", &p_params.aspect);
            changed_p |= ImGui::InputFloat("p_params.zNear", &p_params.zNear);
            changed_p |= ImGui::InputFloat("p_params.zFar", &p_params.zFar);
            if (changed_p) {
                set_perspective_params(camera, p_params);
            }
        } else {
            OrthoParams o_params = get_ortho_params(camera);
            bool changed_o = ImGui::InputFloat("o_params.left", &o_params.left);
            changed_o |= ImGui::InputFloat("o_params.right", &o_params.right);
            changed_o |= ImGui::InputFloat("o_params.bottom", &o_params.bottom);
            changed_o |= ImGui::InputFloat("o_params.top", &o_params.top);
            changed_o |= ImGui::InputFloat("o_params.zNear", &o_params.zNear);
            changed_o |= ImGui::InputFloat("o_params.zFar", &o_params.zFar);
            if (changed_o) {
                set_ortho_params(camera, o_params);
            }
        }
        //TODO set from camera;
        //TODO Convert one to the other ...
        static OrthoParams o_params;
        static PerspectiveParams p_params;
        if (ImGui::RadioButton("Perspective", &projection_type, 0)) {
            if(camera.proj_type == Camera::ProjectionType::ORTHOGRAPHIC){
                o_params = get_ortho_params(camera);
                p_params = Convert(o_params);
            }else{
                p_params = get_perspective_params(camera);
            }
            set_perspective_params(camera, p_params);
        }

        if (ImGui::RadioButton("Orthographic", &projection_type, 1)) {
            if(camera.proj_type == Camera::ProjectionType::PERSPECTIVE){
                p_params = get_perspective_params(camera);
                o_params = Convert(p_params);
            }else{
                o_params = get_ortho_params(camera);
            }
            set_ortho_params(camera, o_params);
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