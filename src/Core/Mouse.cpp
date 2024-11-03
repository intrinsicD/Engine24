//
// Created by alex on 19.06.24.
//

#include "Mouse.h"

namespace Bcg {
    bool Mouse::left() const {
        return pressed[Mouse::ButtonType::Left];
    }

    bool Mouse::middle() const {
        return pressed[Mouse::ButtonType::Middle];
    }

    bool Mouse::right() const {
        return pressed[Mouse::ButtonType::Right];
    }

    bool Mouse::any() const { return left() || middle() || right(); }


}