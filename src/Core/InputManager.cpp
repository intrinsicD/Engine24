//
// Created by alex on 6/15/25.
//
#include "InputManager.h"
#include "PluginInput.h"
#include "Engine.h"
#include "Events/EventsCallbacks.h"
#include "HandleGlfwKeyEvents.h"
#include "imgui.h"
#include <GLFW/glfw3.h>

namespace Bcg {
    InputManager::InputManager() {

    }

    InputManager::~InputManager() {

    }

    void InputManager::handle_key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        auto &keyboard = PluginInput::set_keyboard(window, key, scancode, action, mods);

        if (!keyboard.gui_captured) {
            auto &dispatcher = Engine::Dispatcher();

            dispatcher.trigger<Events::Callback::Key>({window, key, scancode, action, mods});
            if (keyboard.esc()) {
                glfwSetWindowShouldClose(window, true);
            }
            handle(key, action, dispatcher);
        }
    }

    void InputManager::handle_mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
        PluginInput::set_mouse_button(window, button, action, mods);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseButton>({window, button, action, mods});
        }
    }

    void InputManager::handle_mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos) {
        PluginInput::set_mouse_cursor_position(window, xpos, ypos);
        //TODO figure out how to either control the camera or on strg space control the selected object...

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseCursor>({window, xpos, ypos});
        }
    }

    void InputManager::handle_mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        PluginInput::set_mouse_scrolling(window, xoffset, yoffset);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseScroll>({window, xoffset, yoffset});
        }
    }

    void InputManager::handle_window_resize_callback(GLFWwindow *window, int width, int height) {
        Engine::Dispatcher().trigger(Events::Callback::WindowResize{window, width, height});
    }

    void InputManager::handle_framebuffer_resize_callback(GLFWwindow *window, int width, int height) {
        Engine::Dispatcher().trigger(Events::Callback::FramebufferResize{window, width, height});
    }

    void InputManager::handle_close_callback(GLFWwindow *window) {
        glfwSetWindowShouldClose(window, true);

        Engine::Dispatcher().trigger<Events::Callback::WindowClose>({window});
    }

    void InputManager::handle_drop_callback(GLFWwindow *window, int count, const char **paths) {
        for (int i = 0; i < count; ++i) {
            Log::Info("Dropped: {}" , paths[i]);
        }

        Engine::Dispatcher().trigger<Events::Callback::Drop>({window, count, paths});
    }
}
