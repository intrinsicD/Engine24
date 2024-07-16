//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_KEYBOARD_H
#define ENGINE24_KEYBOARD_H

#include <vector>
#include <set>

namespace Bcg {
    struct Keyboard {
        bool shift() const;

        bool strg() const;

        bool alt() const;

        bool esc() const;

        bool w() const;

        bool a() const;

        bool s() const;

        bool d() const;

        bool gui_captured = false;
        std::vector<int> pressed;
        std::set<int> current;
    };

    namespace Gui {
        void Show(const Keyboard &keyboard);

        void Edit(Keyboard &keyboard);
    }
}
#endif //ENGINE24_KEYBOARD_H
