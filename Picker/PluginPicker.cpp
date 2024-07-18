//
// Created by alex on 16.07.24.
//

#include "PluginPicker.h"
#include "PickerGui.h"
#include "Engine.h"
#include "Graphics.h"
#include "Camera.h"
#include "EventsCallbacks.h"
#include "Mouse.h"
#include "imgui.h"

namespace Bcg{

    static void on_construct_entity(entt::registry &registry, entt::entity entity_id){
        Engine::Context().get<Picked>().entity.id = entity_id;
    }

    PluginPicker::PluginPicker() : Plugin("PluginPicker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }

        Engine::State().on_construct<entt::entity>().connect<&on_construct_entity>();
    }

    Picked &PluginPicker::pick(double x, double y) {
        auto &picked = last_picked();
        picked.screen_space_point = {x, y};
        picked.entity.is_background = true;
        float zf;
        if (Graphics::read_depth_buffer(picked.screen_space_point[0], picked.screen_space_point[1], zf)) {
            picked.entity.is_background = false;
            Vector<int, 4> viewport = Graphics::get_viewport();
            picked.ndc_space_point = screen_to_ndc(viewport, picked.screen_space_point[0], picked.screen_space_point[1],
                                                   zf);

            auto &camera = Engine::Context().get<Camera>();

            Matrix<float, 4, 4> vp = camera.proj * camera.view;
            Matrix<float, 4, 4> inv = vp.inverse();
            Vector<float, 4> p = inv * picked.ndc_space_point.homogeneous();
            picked.world_space_point = p.head<3>() / p[3];
            picked.view_space_point = (camera.view * picked.world_space_point.homogeneous()).head<3>();
        }

        return picked;
    }

    Picked &PluginPicker::last_picked() {
        return Engine::Context().get<Picked>();
    }

    static void on_mouse_button(const Events::Callback::MouseButton &event) {
        auto &mouse = Engine::Context().get<Mouse>();
        PluginPicker::pick(mouse.cursor.xpos, mouse.cursor.ypos);
    }


    void PluginPicker::activate() {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
        Engine::Dispatcher().sink<Events::Callback::MouseButton>().connect<&on_mouse_button>();
        Plugin::activate();
    }

    void PluginPicker::begin_frame() {}

    void PluginPicker::update() {}

    void PluginPicker::end_frame() {}

    void PluginPicker::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::MouseButton>().disconnect<&on_mouse_button>();
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginPicker::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginPicker::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = last_picked();
                Gui::Show(picked);
                ImGui::End();
            }
        }
    }

    void PluginPicker::render() {}
}