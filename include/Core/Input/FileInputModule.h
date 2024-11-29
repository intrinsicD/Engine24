//
// Created by alex on 29.11.24.
//

#ifndef ENGINE24_FILEINPUTMODULE_H
#define ENGINE24_FILEINPUTMODULE_H

#include "Module.h"
#include "EventsCallbacks.h"

namespace Bcg{
    class FileInputModule : public Module {
    public:
        FileInputModule();

        ~FileInputModule() override = default;

        void activate() override;

        void deactivate() override;

        static void on_drop(const Events::Callback::Drop &drop);
    };
}

#endif //ENGINE24_FILEINPUTMODULE_H
