//
// Created by alex on 29.11.24.
//

#ifndef ENGINE24_INPUTMODULE_H
#define ENGINE24_INPUTMODULE_H

#include "Module.h"
#include "EventsCallbacks.h"

namespace Bcg{
    class InputModule : public Module {
    public:
        InputModule();

        ~InputModule() override = default;

        void activate() override;

        void deactivate() override;
    };
}

#endif //ENGINE24_INPUTMODULE_H
