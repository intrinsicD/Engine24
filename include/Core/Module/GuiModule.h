//
// Created by alex on 25.11.24.
//

#ifndef GUIMODULE_H
#define GUIMODULE_H

#include "Module.h"

namespace Bcg {
    class GuiModule : public Module {
    public:
        explicit GuiModule(const std::string &name) : Module(name) {}

        ~GuiModule() override = default;
    };
}

#endif //GUIMODULE_H
