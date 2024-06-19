//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_KEYBAORD_H
#define ENGINE24_KEYBAORD_H

#include <vector>

namespace Bcg {
    struct Keyboard {
        struct Key {
            const char *name;
            int key;
            int scancode;
            int action;
            int mode;

            operator bool() const { return action; }
        };

        Key shift;
        Key strg;
        Key alt;
        Key esc;

        std::vector<unsigned int> pressed;
    };
}
#endif //ENGINE24_KEYBAORD_H
