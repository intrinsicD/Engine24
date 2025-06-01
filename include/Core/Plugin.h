//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGIN_H
#define ENGINE24_PLUGIN_H

#include "GuiModule.h"

namespace Bcg {
    class Plugin : public GuiModule{
    public:
        explicit Plugin(const std::string &name) : GuiModule(name) {}

        virtual ~Plugin() = default;
    };
}

#endif //ENGINE24_PLUGIN_H
