//
// Created by alex on 19.06.24.
//

#include "Input.h"
#include "Engine.h"
#include "Keyboard.h"
#include "Mouse.h"
#include "Camera.h"
#include "Graphics.h"
#include "imgui.h"
#include "EventsKeys.h"

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


    Keyboard &Input::set_keyboard(GLFWwindow *window, int key, int scancode, int action, int mode) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (keyboard.gui_captured) return keyboard;
        while (key >= keyboard.pressed.size()) {
            keyboard.pressed.emplace_back(0);
        }
        keyboard.pressed[key] = action;
        if (action) {
            keyboard.current.emplace(key);
        } else {
            keyboard.current.erase(key);
        }
        return keyboard;
    }

    Mouse &Input::set_mouse_cursor_position(GLFWwindow *window, double xpos, double ypos) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (mouse.gui_captured) return mouse;
        auto &camera = Engine::Context().get<Camera>();
        float zf;
        Graphics::read_depth_buffer(xpos, ypos, zf);
        mouse.cursor.current = PointTransformer(Graphics::dpi_scaling(), Graphics::get_viewport_dpi_adjusted(),
                                                camera.proj,
                                                camera.view).apply(ScreenSpacePos(xpos, ypos), zf);
        return mouse;
    }

    Mouse &Input::set_mouse_button(GLFWwindow *window, int button, int action, int mods) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (mouse.gui_captured) return mouse;
        while (button >= mouse.pressed.size()) {
            mouse.pressed.emplace_back(0);
        }
        mouse.pressed[button] = action;
        if (action) {
            mouse.current_buttons.emplace(button);
        } else {
            mouse.current_buttons.erase(button);
        }
        if (action) {
            //press
            if (button == Mouse::ButtonType::Left) {
                mouse.cursor.last_left.press = mouse.cursor.current;
            }
            if (button == Mouse::ButtonType::Middle) {
                mouse.cursor.last_middle.press = mouse.cursor.current;
            }
            if (button == Mouse::ButtonType::Right) {
                mouse.cursor.last_right.press = mouse.cursor.current;
            }
        } else {
            //release
            if (button == Mouse::ButtonType::Left) {
                mouse.cursor.last_left.release = mouse.cursor.current;
            }
            if (button == Mouse::ButtonType::Middle) {
                mouse.cursor.last_middle.release = mouse.cursor.current;
            }
            if (button == Mouse::ButtonType::Right) {
                mouse.cursor.last_right.release = mouse.cursor.current;
            }
        }
        return mouse;
    }

    Mouse &Input::set_mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (mouse.gui_captured) return mouse;
        mouse.scrolling = true;
        return mouse;
    }

    void Input::activate() {
        Plugin::activate();
    }

    void Input::begin_frame() {
        auto &mouse = Engine::Context().get<Mouse>();
        mouse.gui_captured = ImGui::GetIO().WantCaptureMouse;

        auto &keyboard = Engine::Context().get<Keyboard>();
        keyboard.gui_captured = ImGui::GetIO().WantCaptureKeyboard;
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
                Gui::Show(Engine::Context().get<Keyboard>());
                ImGui::Separator();
                ImGui::Text("Mouse:");
                ImGui::Separator();
                Gui::Show(Engine::Context().get<Mouse>());
            }
            ImGui::End();
        }
    }

    void Input::render() {

    }
}