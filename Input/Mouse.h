//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_MOUSE_H
#define ENGINE24_MOUSE_H

#include <vector>
#include <set>
#include "GuiUtils.h"
#include "MatVec.h"

namespace Bcg {
    struct Mouse {
        struct Cursor {
            union{
                Vector<float, 2> screen_space;
                float xpos, ypos;
            };
        };

        bool left() const;

        bool middle() const;

        bool right() const;

        bool any() const;

        bool scrolling = false;

        bool gui_captured = false;
        std::vector<int> pressed;
        std::set<int> current;

        Cursor cursor;
    };


    namespace Gui {
        void Show(const Mouse &mouse);

        void Show(const Mouse::Cursor &cursor);
    };
}

#endif //ENGINE24_MOUSE_H
