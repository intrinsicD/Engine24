//
// Created by alex on 15.07.24.
//

#include "PluginTransform.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "Engine.h"
#include "EventsGui.h"
#include "Picker.h"
#include "Camera.h"
#include "Graphics.h"

namespace Bcg {
    PluginTransform::PluginTransform() : Plugin("Transform") {}

    void PluginTransform::activate() {
        Plugin::activate();
    }

    void PluginTransform::begin_frame() {

    }

    void PluginTransform::update() {

    }

    void PluginTransform::end_frame() {

    }

    void PluginTransform::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    static void on_gui_render(const Events::Gui::Render &event) {
        if (!show_gui) {
            Engine::Dispatcher().sink<Events::Gui::Render>().disconnect<on_gui_render>();
            return;
        }

        auto &picked = Bcg::Picker::last_picked();
        Gui::ShowTransform(picked.entity.id);
    }

    void PluginTransform::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::MenuItem(name, nullptr, &show_gui)) {
                Engine::Dispatcher().sink<Events::Gui::Render>().connect<on_gui_render>();
            }
            ImGui::EndMenu();
        }
    }

    void PluginTransform::render_gui() {


    }

    void PluginTransform::render() {

    }

    namespace Gui {
        static bool enable_guizmo = false;

        void ShowTransform(entt::entity entity_id) {
            if (ImGui::Begin("Transform", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (Engine::valid(entity_id) && Engine::has<Transform>(entity_id)) {
                    ImGui::Checkbox("Guizmo", &enable_guizmo);
                    if (enable_guizmo) {

                        auto &transform = Engine::State().get<Transform>(entity_id);
                        auto &camera = Engine::Context().get<Camera>();
                        auto &view = camera.view;
                        auto &projection = camera.proj;

                        // Define operation and mode (translate, rotate, scale)
                        static ImGuizmo::OPERATION currentOperation(ImGuizmo::TRANSLATE);
                        static ImGuizmo::MODE currentMode(ImGuizmo::WORLD);

                        // Show the Gizmo and handle interactions
                        ImGuiIO& io = ImGui::GetIO();
                        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
                        ImGuizmo::Manipulate(view.data(), projection.data(), currentOperation, currentMode, transform.data());
                    }
                    Show(Engine::State().get<Transform>(entity_id));
                }
            }
            ImGui::End();
        }

        void Show(Transform &transform) {
            //decompose the 4, 4 matrix into translation, rotations and scaling and let them be controlled by the user like ImGuizmo does it
            Eigen::Vector3<float> translation, scaling;
            Eigen::Matrix3<float> rotation;
            ImGuizmo::DecomposeMatrixToComponents(transform.data(), translation.data(), rotation.data(), scaling.data());
            ImGui::InputFloat3("Tr", translation.data());
            ImGui::InputFloat3("Rt", rotation.data());
            ImGui::InputFloat3("Sc", scaling.data());
            ImGuizmo::RecomposeMatrixFromComponents(translation.data(), rotation.data(), scaling.data(), transform.data());
        }
    }
}