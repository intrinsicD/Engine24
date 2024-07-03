//
// Created by alex on 03.07.24.
//

#include "Picker.h"
#include "Engine.h"

namespace Bcg {
    Picker::Picker() : Plugin("Picker") {
        if (!Engine::Context().find<Picked>()) {
            Engine::Context().emplace<Picked>();
        }
    }

    Picked &Picker::pick(double x, double y) {
        return last_picked();
    }

    Picked &Picker::last_picked() {
        return Engine::Context().get<Picked &>();
    }
}