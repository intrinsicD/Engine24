//
// Created by alex on 29.11.24.
//

#ifndef ENGINE24_INPUTMODULE_H
#define ENGINE24_INPUTMODULE_H

#include "Module.h"
#include "EventsCallbacks.h"
#include "EventsMain.h"

namespace Bcg{
    class InputModule : public Module {
    public:
        InputModule();

        ~InputModule() override = default;

        void activate() override;

        void deactivate() override;

        static void on_synchronize(const Events::Synchronize &event);

        static void on_drop(const Events::Callback::Drop &event);

        static void on_key(const Events::Callback::Key &event);

        static void on_mouse_button(const Events::Callback::MouseButton &event);

        static void on_mouse_cursor(const Events::Callback::MouseCursor &event);

        static void on_mouse_scroll(const Events::Callback::MouseScroll &event);

        static void on_framebuffer_resize(const Events::Callback::FramebufferResize &event);

        static void on_window_resize(const Events::Callback::WindowResize &event);

        static void on_window_close(const Events::Callback::WindowClose &event);
    };
}

#endif //ENGINE24_INPUTMODULE_H
