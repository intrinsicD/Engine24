//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_MOUSE_H
#define ENGINE24_MOUSE_H

#include <vector>
#include <set>
#include "GuiUtils.h"
#include "CoordinateSystems.h"

namespace Bcg {
    struct Mouse {
        enum ButtonType{
            Left = 0,
            Right = 1,
            Middle = 2
        };
        struct Cursor {
            Points current;
            struct {
                Points press;
                Points release;
            }last_left;
            struct {
                Points press;
                Points release;
            }last_middle;
            struct {
                Points press;
                Points release;
            }last_right;
        };

        bool left() const;

        bool middle() const;

        bool right() const;

        bool any() const;

        bool scrolling = false;

        bool gui_captured = false;
        std::vector<int> pressed;
        std::set<int> current_buttons;

        Cursor cursor;
    };


    namespace Gui {
        void Show(const Mouse &mouse);
        void Show(const Mouse &mouse);

        void Show(const Mouse::Cursor &cursor);
    };
}

#endif //ENGINE24_MOUSE_H
