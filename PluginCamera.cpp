//
// Created by alex on 08.07.24.
//

#include "PluginCamera.h"
#include "imgui.h"

namespace Bcg {
    PluginCamera::PluginCamera() : Plugin("Camera") {}

    void PluginCamera::activate() {
        Plugin::activate();
    }

    void PluginCamera::begin_frame() {

    }

    void PluginCamera::update() {

    }

    void PluginCamera::end_frame() {

    }

    void PluginCamera::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginCamera::render_menu() {
        if (ImGui::BeginMenu(name)) {
            if (ImGui::MenuItem("Camera", nullptr, &show_gui)) {
                activate();
            }
            ImGui::EndMenu();
        }
    }

    void PluginCamera::render_gui(Camera &camera) {
        if (show_gui) {
            if (ImGui::Begin("Camera", &show_gui)) {
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
                    ImGui::InputFloat("p_params.fovy", &camera.p_params.fovy);
                    ImGui::InputFloat("p_params.aspect", &camera.p_params.aspect);
                    ImGui::InputFloat("p_params.zNear", &camera.p_params.zNear);
                    ImGui::InputFloat("p_params.zFar", &camera.p_params.zFar);
                } else {
                    ImGui::InputFloat("o_params.left", &camera.o_params.left);
                    ImGui::InputFloat("o_params.right", &camera.o_params.right);
                    ImGui::InputFloat("o_params.bottom", &camera.o_params.bottom);
                    ImGui::InputFloat("o_params.top", &camera.o_params.top);
                    ImGui::InputFloat("o_params.zNear", &camera.o_params.zNear);
                    ImGui::InputFloat("o_params.zFar", &camera.o_params.zFar);
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
                    camera.v_params = Camera::ViewParameters();
                    camera.p_params = Camera::PerspParameters();
                    camera.o_params = Camera::OrthoParameters();
                    camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
                    camera.view = look_at_matrix(camera.v_params.eye, camera.v_params.center, camera.v_params.up);
                    camera.proj = perspective_matrix(camera.p_params.fovy, camera.p_params.aspect,
                                                     camera.p_params.zNear,
                                                     camera.p_params.zFar);
                }
                ImGui::End();
            }
        }
    }

    void PluginCamera::render_gui() {

    }

    void PluginCamera::render() {

    }
}