//
// Created by alex on 29.11.24.
//

#include "InputModule.h"

namespace Bcg {

    InputModule::InputModule() : Module("InputModule") {}

    void InputModule::activate() {
        base_activate();
    }

    void InputModule::deactivate() {
        base_deactivate();
    }
}