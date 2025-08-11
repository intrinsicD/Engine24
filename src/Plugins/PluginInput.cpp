//
// Created by alex on 19.06.24.
//

#include "PluginInput.h"
#include "Engine.h"
#include "Camera.h"
#include "../../Graphics/ModuleGraphics.h"
#include "MouseGui.h"
#include "imgui.h"
#include "Application.h"

namespace Bcg {
    PluginInput::PluginInput() : Plugin("Input") {
        if (!Engine::Context().find<Keyboard>()) {
            auto &keyboard = Engine::Context().emplace<Keyboard>();
            keyboard.pressed.resize(1024);
        }
        if (!Engine::Context().find<Mouse>()) {
            auto &mouse = Engine::Context().emplace<Mouse>();
            mouse.pressed.resize(5);
        }
    }


    Keyboard &PluginInput::set_keyboard(GLFWwindow *window, int key, int scancode, int action, int mode) {
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

    Mouse &PluginInput::set_mouse_cursor_position(GLFWwindow */*window*/, double xpos, double ypos) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (mouse.gui_captured) return mouse;
        auto &camera = Engine::Context().get<Camera>();
        float zf;
        ModuleGraphics::read_depth_buffer(xpos, ypos, zf);
        auto *window = Engine::Context().get<Application*>()->window.get();
        auto *renderer = Engine::Context().get<Application *>()->renderer.get();
        float xscale = window->get_xy_dpi_scaling()[0];
        mouse.cursor.current = PointTransformer(xscale,renderer->get_viewport_dpi_adjusted(),
                                                camera.proj,
                                                camera.view).apply(ScreenSpacePos(xpos, ypos), zf);
        return mouse;
    }

    Mouse &PluginInput::set_mouse_button(GLFWwindow *window, int button, int action, int mods) {
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

    Mouse &PluginInput::set_mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (mouse.gui_captured) return mouse;
        mouse.scroll_offset = {xoffset, yoffset};
        mouse.scrolling = true;
        return mouse;
    }

    void PluginInput::activate() {
        if (base_activate()) {

        }
    }

    void PluginInput::begin_frame() {
        auto &mouse = Engine::Context().get<Mouse>();
        mouse.gui_captured = ImGui::GetIO().WantCaptureMouse;

        auto &keyboard = Engine::Context().get<Keyboard>();
        keyboard.gui_captured = ImGui::GetIO().WantCaptureKeyboard;
    }

    void PluginInput::update() {

    }

    void PluginInput::end_frame() {
        auto &mouse = Engine::Context().get<Mouse>();
        mouse.scrolling = false;
        mouse.scroll_offset = glm::vec2(0.0f);
    }

    void PluginInput::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool show_input_gui;

    void PluginInput::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem("Input", nullptr, &show_input_gui);
            ImGui::EndMenu();
        }
    }

    void PluginInput::render_gui() {
        if (show_input_gui) {
            if (ImGui::Begin("Module", &show_input_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
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

    void PluginInput::render() {

    }
}