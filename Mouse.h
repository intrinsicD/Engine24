//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_MOUSE_H
#define ENGINE24_MOUSE_H

#include <vector>

namespace Bcg {
    struct Mouse {
        struct Button {
            int button;
            int action;
            int mods;

            operator bool() const { return action; }
        };

        struct Cursor {
            double xpos, ypos;
        };

        Button left;
        Button middle;
        Button right;
        Cursor cursor;

        bool any() const { return left || middle || right; }
    };
}

#endif //ENGINE24_MOUSE_H
