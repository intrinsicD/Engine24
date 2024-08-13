//
// Created by alex on 10.07.24.
//

#include "Engine.h"
#include "HandleGlfwKeyEvents.h"
#include "EventsKeys.h"
#include "GLFW/glfw3.h"

namespace Bcg {
    using TriggerFunction = void(*)(int, entt::dispatcher &); // pointer to the trigger function

    static std::unordered_map<int, TriggerFunction> keyTrigger_map {
            { GLFW_KEY_A, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::A{action}); }},
            { GLFW_KEY_B, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::B{action}); }},
            { GLFW_KEY_C, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::C{action}); }},
            { GLFW_KEY_D, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::D{action}); }},
            { GLFW_KEY_E, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::E{action}); }},
            { GLFW_KEY_F, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F{action}); }},
            { GLFW_KEY_G, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::G{action}); }},
            { GLFW_KEY_H, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::H{action}); }},
            { GLFW_KEY_I, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::I{action}); }},
            { GLFW_KEY_J, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::J{action}); }},
            { GLFW_KEY_K, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::K{action}); }},
            { GLFW_KEY_L, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::L{action}); }},
            { GLFW_KEY_M, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::M{action}); }},
            { GLFW_KEY_N, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::N{action}); }},
            { GLFW_KEY_O, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::O{action}); }},
            { GLFW_KEY_P, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::P{action}); }},
            { GLFW_KEY_Q, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Q{action}); }},
            { GLFW_KEY_R, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::R{action}); }},
            { GLFW_KEY_S, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::S{action}); }},
            { GLFW_KEY_T, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::T{action}); }},
            { GLFW_KEY_U, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::U{action}); }},
            { GLFW_KEY_V, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::V{action}); }},
            { GLFW_KEY_W, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::W{action}); }},
            { GLFW_KEY_X, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::X{action}); }},
            { GLFW_KEY_Y, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Y{action}); }},
            { GLFW_KEY_Z, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Z{action}); }},
            { GLFW_KEY_0, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Zero{action}); }},
            { GLFW_KEY_1, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::One{action}); }},
            { GLFW_KEY_2, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Two{action}); }},
            { GLFW_KEY_3, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Three{action}); }},
            { GLFW_KEY_4, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Four{action}); }},
            { GLFW_KEY_5, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Five{action}); }},
            { GLFW_KEY_6, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Six{action}); }},
            { GLFW_KEY_7, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Seven{action}); }},
            { GLFW_KEY_8, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Eight{action}); }},
            { GLFW_KEY_9, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Nine{action}); }},
            { GLFW_KEY_ESCAPE, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Esc{action}); }},
            { GLFW_KEY_SPACE, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Space{action}); }},
            { GLFW_KEY_ENTER, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Enter{action}); }},
            { GLFW_KEY_BACKSPACE, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Backspace{action}); }},
            { GLFW_KEY_DELETE, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Delete{action}); }},
            { GLFW_KEY_LEFT_SHIFT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Shift{action}); }},
            { GLFW_KEY_RIGHT_SHIFT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Shift{action}); }},
            { GLFW_KEY_LEFT_CONTROL, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Ctrl{action}); }},
            { GLFW_KEY_RIGHT_CONTROL, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Ctrl{action}); }},
            { GLFW_KEY_LEFT_ALT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Alt{action}); }},
            { GLFW_KEY_RIGHT_ALT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Alt{action}); }},
            { GLFW_KEY_TAB, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Tab{action}); }},
            { GLFW_KEY_CAPS_LOCK, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::CapsLock{action}); }},
            { GLFW_KEY_UP, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::ArrowUp{action}); }},
            { GLFW_KEY_DOWN, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::ArrowDown{action}); }},
            { GLFW_KEY_LEFT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::ArrowLeft{action}); }},
            { GLFW_KEY_RIGHT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::ArrowRight{action}); }},
            { GLFW_KEY_HOME, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Home{action}); }},
            { GLFW_KEY_END, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::End{action}); }},
            { GLFW_KEY_PAGE_UP, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::PageUp{action}); }},
            { GLFW_KEY_PAGE_DOWN, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::PageDown{action}); }},
            { GLFW_KEY_INSERT, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::Insert{action}); }},
            { GLFW_KEY_F1, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F1{action}); }},
            { GLFW_KEY_F2, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F2{action}); }},
            { GLFW_KEY_F3, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F3{action}); }},
            { GLFW_KEY_F4, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F4{action}); }},
            { GLFW_KEY_F5, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F5{action}); }},
            { GLFW_KEY_F6, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F6{action}); }},
            { GLFW_KEY_F7, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F7{action}); }},
            { GLFW_KEY_F8, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F8{action}); }},
            { GLFW_KEY_F9, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F9{action}); }},
            { GLFW_KEY_F10, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F10{action}); }},
            { GLFW_KEY_F11, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F11{action}); }},
            { GLFW_KEY_F12, [](int action, entt::dispatcher &dispatcher){ dispatcher.trigger(Events::Key::F12{action}); }},
    };

    void handle(int key, int action, entt::dispatcher &dispatcher) {
        if (keyTrigger_map.count(key)){
            keyTrigger_map[key](action, dispatcher);
        }
    }
}