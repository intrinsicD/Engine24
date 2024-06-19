//
// Created by alex on 19.06.24.
//

#include "Input.h"
#include "Engine.h"
#include "Keybaord.h"
#include "Mouse.h"
#include "imgui.h"

namespace Bcg {
    static bool show_input_gui;

    Input::Input() : Plugin("Input") {
        if (!Engine::Context().find<Keyboard>()) {
            auto &keyboard = Engine::Context().emplace<Keyboard>();
            keyboard.pressed.resize(1024);
        }
        if (!Engine::Context().find<Mouse>()) {
            auto &mouse = Engine::Context().emplace<Mouse>();
        }
    }

    void Input::activate() {
        Plugin::activate();
    }

    void Input::begin_frame() {

    }

    void Input::update() {
        auto &keyboard = Engine::Context().get<Keyboard>();
        auto &mouse = Engine::Context().get<Mouse>();
    }

    void Input::end_frame() {
        auto &keyboard = Engine::Context().get<Keyboard>();
        auto &mouse = Engine::Context().get<Mouse>();
    }

    void Input::deactivate() {
        Plugin::deactivate();
    }

    void Input::render_menu() {
        if (ImGui::BeginMenu("Input")) {
            ImGui::MenuItem("Input", nullptr, &show_input_gui);
            ImGui::EndMenu();
        }
    }

    void Input::render_gui() {
        if (show_input_gui) {
            if (ImGui::Begin("Input", &show_input_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                render_gui(Engine::Context().get<Keyboard>());
                ImGui::Separator();
                render_gui(Engine::Context().get<Mouse>());
            }
            ImGui::End();
        }
    }

    void Input::render_gui(const Keyboard::Key &key) {
        ImGui::Text("%s: %d Scancode: %d Action: %d Mode: %d", key.name, key.key, key.scancode,
                    key.action, key.mode);
    }

    void Input::render_gui(const Keyboard &keyboard) {
        render_gui(keyboard.shift);
        ImGui::Separator();
        render_gui(keyboard.strg);
        ImGui::Separator();
        render_gui(keyboard.alt);
        ImGui::Separator();
        render_gui(keyboard.esc);
    }

    void Input::render_gui(const Mouse::Button &button) {
        ImGui::Text("Button: %d Action: %d Mods: %d", button.button, button.action, button.mods);
    }

    void Input::render_gui(const Mouse::Cursor &cursor) {
        ImGui::Text("Position: %lf, %lf", cursor.xpos, cursor.ypos);
    }

    void Input::render_gui(const Mouse &mouse) {
        render_gui(mouse.left);
        ImGui::Separator();
        render_gui(mouse.right);
        ImGui::Separator();
        render_gui(mouse.middle);
        ImGui::Separator();
        render_gui(mouse.cursor);
    }

    void Input::render() {

    }
}