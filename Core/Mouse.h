//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_MOUSE_H
#define ENGINE24_MOUSE_H

#include <vector>
#include <set>

namespace Bcg {
    struct Mouse {
        struct Cursor {
            double xpos, ypos;
        };

        bool left() const;

        bool middle() const;

        bool right() const;

        bool any() const;

        bool scrolling = false;

        std::vector<int> pressed;
        std::set<int> current;

        Cursor cursor;
    };
}

#endif //ENGINE24_MOUSE_H
