//
// Created by alex on 10.07.24.
//

#include "Engine.h"
#include "HandleKeyEvents.h"
#include "EventsKeys.h"
#include "GLFW/glfw3.h"

namespace Bcg {
    void handle(int key, int action) {
        auto &dispatcher = Engine::Dispatcher();
        switch (key) {
            case GLFW_KEY_A:
                dispatcher.trigger<Bcg::Events::Key::A>(action);
                break;
            case GLFW_KEY_B:
                dispatcher.trigger<Bcg::Events::Key::B>(action);
                break;
            case GLFW_KEY_C:
                dispatcher.trigger<Bcg::Events::Key::C>(action);
                break;
            case GLFW_KEY_D:
                dispatcher.trigger<Bcg::Events::Key::D>(action);
                break;
            case GLFW_KEY_E:
                dispatcher.trigger<Bcg::Events::Key::E>(action);
                break;
            case GLFW_KEY_F:
                dispatcher.trigger<Bcg::Events::Key::F>(action);
                break;
            case GLFW_KEY_G:
                dispatcher.trigger<Bcg::Events::Key::G>(action);
                break;
            case GLFW_KEY_H:
                dispatcher.trigger<Bcg::Events::Key::H>(action);
                break;
            case GLFW_KEY_I:
                dispatcher.trigger<Bcg::Events::Key::I>(action);
                break;
            case GLFW_KEY_J:
                dispatcher.trigger<Bcg::Events::Key::J>(action);
                break;
            case GLFW_KEY_K:
                dispatcher.trigger<Bcg::Events::Key::K>(action);
                break;
            case GLFW_KEY_L:
                dispatcher.trigger<Bcg::Events::Key::L>(action);
                break;
            case GLFW_KEY_M:
                dispatcher.trigger<Bcg::Events::Key::M>(action);
                break;
            case GLFW_KEY_N:
                dispatcher.trigger<Bcg::Events::Key::N>(action);
                break;
            case GLFW_KEY_O:
                dispatcher.trigger<Bcg::Events::Key::O>(action);
                break;
            case GLFW_KEY_P:
                dispatcher.trigger<Bcg::Events::Key::P>(action);
                break;
            case GLFW_KEY_Q:
                dispatcher.trigger<Bcg::Events::Key::Q>(action);
                break;
            case GLFW_KEY_R:
                dispatcher.trigger<Bcg::Events::Key::R>(action);
                break;
            case GLFW_KEY_S:
                dispatcher.trigger<Bcg::Events::Key::S>(action);
                break;
            case GLFW_KEY_T:
                dispatcher.trigger<Bcg::Events::Key::T>(action);
                break;
            case GLFW_KEY_U:
                dispatcher.trigger<Bcg::Events::Key::U>(action);
                break;
            case GLFW_KEY_V:
                dispatcher.trigger<Bcg::Events::Key::V>(action);
                break;
            case GLFW_KEY_W:
                dispatcher.trigger<Bcg::Events::Key::W>(action);
                break;
            case GLFW_KEY_X:
                dispatcher.trigger<Bcg::Events::Key::X>(action);
                break;
            case GLFW_KEY_Y:
                dispatcher.trigger<Bcg::Events::Key::Y>(action);
                break;
            case GLFW_KEY_Z:
                dispatcher.trigger<Bcg::Events::Key::Z>(action);
                break;
            case GLFW_KEY_0:
                dispatcher.trigger<Bcg::Events::Key::Zero>(action);
                break;
            case GLFW_KEY_1:
                dispatcher.trigger<Bcg::Events::Key::One>(action);
                break;
            case GLFW_KEY_2:
                dispatcher.trigger<Bcg::Events::Key::Two>(action);
                break;
            case GLFW_KEY_3:
                dispatcher.trigger<Bcg::Events::Key::Three>(action);
                break;
            case GLFW_KEY_4:
                dispatcher.trigger<Bcg::Events::Key::Four>(action);
                break;
            case GLFW_KEY_5:
                dispatcher.trigger<Bcg::Events::Key::Five>(action);
                break;
            case GLFW_KEY_6:
                dispatcher.trigger<Bcg::Events::Key::Six>(action);
                break;
            case GLFW_KEY_7:
                dispatcher.trigger<Bcg::Events::Key::Seven>(action);
                break;
            case GLFW_KEY_8:
                dispatcher.trigger<Bcg::Events::Key::Eight>(action);
                break;
            case GLFW_KEY_9:
                dispatcher.trigger<Bcg::Events::Key::Nine>(action);
                break;
            case GLFW_KEY_ESCAPE:
                dispatcher.trigger<Bcg::Events::Key::Esc>(action);
                break;
            case GLFW_KEY_SPACE:
                dispatcher.trigger<Bcg::Events::Key::Space>(action);
                break;
            case GLFW_KEY_ENTER:
                dispatcher.trigger<Bcg::Events::Key::Enter>(action);
                break;
            case GLFW_KEY_BACKSPACE:
                dispatcher.trigger<Bcg::Events::Key::Backspace>(action);
                break;
            case GLFW_KEY_DELETE:
                dispatcher.trigger<Bcg::Events::Key::Delete>(action);
                break;
            case GLFW_KEY_LEFT_SHIFT:
            case GLFW_KEY_RIGHT_SHIFT:
                dispatcher.trigger<Bcg::Events::Key::Shift>(action);
                break;
            case GLFW_KEY_LEFT_CONTROL:
            case GLFW_KEY_RIGHT_CONTROL:
                dispatcher.trigger<Bcg::Events::Key::Ctrl>(action);
                break;
            case GLFW_KEY_LEFT_ALT:
            case GLFW_KEY_RIGHT_ALT:
                dispatcher.trigger<Bcg::Events::Key::Alt>(action);
                break;
            case GLFW_KEY_TAB:
                dispatcher.trigger<Bcg::Events::Key::Tab>(action);
                break;
            case GLFW_KEY_CAPS_LOCK:
                dispatcher.trigger<Bcg::Events::Key::CapsLock>(action);
                break;
            case GLFW_KEY_UP:
                dispatcher.trigger<Bcg::Events::Key::ArrowUp>(action);
                break;
            case GLFW_KEY_DOWN:
                dispatcher.trigger<Bcg::Events::Key::ArrowDown>(action);
                break;
            case GLFW_KEY_LEFT:
                dispatcher.trigger<Bcg::Events::Key::ArrowLeft>(action);
                break;
            case GLFW_KEY_RIGHT:
                dispatcher.trigger<Bcg::Events::Key::ArrowRight>(action);
                break;
            case GLFW_KEY_HOME:
                dispatcher.trigger<Bcg::Events::Key::Home>(action);
                break;
            case GLFW_KEY_END:
                dispatcher.trigger<Bcg::Events::Key::End>(action);
                break;
            case GLFW_KEY_PAGE_UP:
                dispatcher.trigger<Bcg::Events::Key::PageUp>(action);
                break;
            case GLFW_KEY_PAGE_DOWN:
                dispatcher.trigger<Bcg::Events::Key::PageDown>(action);
                break;
            case GLFW_KEY_INSERT:
                dispatcher.trigger<Bcg::Events::Key::Insert>(action);
                break;
            case GLFW_KEY_F1:
                dispatcher.trigger<Bcg::Events::Key::F1>(action);
                break;
            case GLFW_KEY_F2:
                dispatcher.trigger<Bcg::Events::Key::F2>(action);
                break;
            case GLFW_KEY_F3:
                dispatcher.trigger<Bcg::Events::Key::F3>(action);
                break;
            case GLFW_KEY_F4:
                dispatcher.trigger<Bcg::Events::Key::F4>(action);
                break;
            case GLFW_KEY_F5:
                dispatcher.trigger<Bcg::Events::Key::F5>(action);
                break;
            case GLFW_KEY_F6:
                dispatcher.trigger<Bcg::Events::Key::F6>(action);
                break;
            case GLFW_KEY_F7:
                dispatcher.trigger<Bcg::Events::Key::F7>(action);
                break;
            case GLFW_KEY_F8:
                dispatcher.trigger<Bcg::Events::Key::F8>(action);
                break;
            case GLFW_KEY_F9:
                dispatcher.trigger<Bcg::Events::Key::F9>(action);
                break;
            case GLFW_KEY_F10:
                dispatcher.trigger<Bcg::Events::Key::F10>(action);
                break;
            case GLFW_KEY_F11:
                dispatcher.trigger<Bcg::Events::Key::F11>(action);
                break;
            case GLFW_KEY_F12:
                dispatcher.trigger<Bcg::Events::Key::F12>(action);
                break;
            default:
                break;
        }
    }
}