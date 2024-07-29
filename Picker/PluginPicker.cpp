//
// Created by alex on 16.07.24.
//

#include "PluginPicker.h"
#include "PickerGui.h"
#include "Engine.h"
#include "Graphics.h"
#include "EventsCallbacks.h"
#include "Mouse.h"
#include "imgui.h"
#include "Intersections.h"
#include "Transform.h"

namespace Bcg {

    static void on_construct_entity(entt::registry &registry, entt::entity entity_id) {
        Engine::Context().get<Picked>().entity.id = entity_id;
    }

    PluginPicker::PluginPicker() : Plugin("PluginPicker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }

        Engine::State().on_construct<entt::entity>().connect<&on_construct_entity>();
    }

    Picked &PluginPicker::pick(const ScreenSpacePos &pos) {
        auto &mouse = Engine::Context().get<Mouse>();
        auto &picked = last_picked();
        picked.spaces = mouse.cursor.last_left.press;
        picked.entity.is_background = picked.spaces.ndc.z() == 1.0;
        auto view = Engine::State().view<AABB, Transform>();
        for (const auto entity_id: view) {
            auto &aabb = Engine::State().get<AABB>(entity_id);
            auto &transform = Engine::State().get<Transform>(entity_id);
            if (Intersect(aabb, (transform.world().inverse() * picked.spaces.wsp.homogeneous()).head<3>())) {
                picked.entity.id = entity_id;
                return picked;
            }
        }
        return picked;
    }

    Picked &PluginPicker::last_picked() {
        return Engine::Context().get<Picked>();
    }

    static void on_mouse_button(const Events::Callback::MouseButton &event) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (event.action) {
            PluginPicker::pick(mouse.cursor.current.ssp);
        }
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