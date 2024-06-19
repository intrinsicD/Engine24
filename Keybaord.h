//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_KEYBAORD_H
#define ENGINE24_KEYBAORD_H

#include <vector>
#include <set>

namespace Bcg {
    struct Keyboard {
        bool shift() const;

        bool strg() const;

        bool alt() const;

        bool esc() const;

        std::vector<int> pressed;
        std::set<int> current;
    };
}
#endif //ENGINE24_KEYBAORD_H
