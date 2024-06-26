//
// Created by alex on 19.06.24.
//

#include "Input.h"
#include "Engine.h"
#include "Keybaord.h"
#include "Mouse.h"
#include "imgui.h"
#include "GLFWUtils.h"

namespace Bcg {
    static bool show_input_gui;

    Input::Input() : Plugin("Input") {
        if (!Engine::Context().find<Keyboard>()) {
            auto &keyboard = Engine::Context().emplace<Keyboard>();
            keyboard.pressed.resize(1024);
        }
        if (!Engine::Context().find<Mouse>()) {
            auto &mouse = Engine::Context().emplace<Mouse>();
            mouse.pressed.resize(5);
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
        mouse.scrolling = false;
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
                ImGui::Text("Keyboard:");
                ImGui::Separator();
                render_gui(Engine::Context().get<Keyboard>());
                ImGui::Separator();
                ImGui::Text("Mouse:");
                ImGui::Separator();
                render_gui(Engine::Context().get<Mouse>());
            }
            ImGui::End();
        }
    }

    void Input::render_gui(const Keyboard &keyboard) {
        ImGui::Text("Shift: %d", keyboard.shift());
        ImGui::Text("Strg: %d", keyboard.strg());
        ImGui::Text("Alt: %d", keyboard.alt());
        ImGui::Text("Esc: %d", keyboard.esc());
        ImGui::Text("Current Keys: {");
        ImGui::SameLine();
        for (const auto key: keyboard.current) {
            ImGui::Text("%s", KeyName(key));
            ImGui::SameLine();
        }
        ImGui::Text("}");
    }

    void Input::render_gui(const Mouse::Cursor &cursor) {
        ImGui::Text("Position: %lf, %lf", cursor.xpos, cursor.ypos);
    }

    void Input::render_gui(const Mouse &mouse) {
        ImGui::Text("Left: %d", mouse.left());
        ImGui::Text("Middle: %d", mouse.middle());
        ImGui::Text("Right: %d", mouse.right());
        ImGui::Text("Scrolling: %d", mouse.scrolling);
        render_gui(mouse.cursor);
        ImGui::Text("Current Buttons: {");
        ImGui::SameLine();
        for (const auto button: mouse.current) {
            ImGui::Text("%d", button);
            ImGui::SameLine();
        }
        ImGui::Text("}");
    }

    void Input::render() {

    }
}