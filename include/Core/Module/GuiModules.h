//
// Created by alex on 25.11.24.
//

#ifndef GUIMODULES_H
#define GUIMODULES_H

#include <memory>
#include <unordered_map>
#include "GuiModule.h"

namespace Bcg {
    class GuiModules {
    public:
        void activate();

        void deactivate();

        void add(const std::string &name, std::unique_ptr<GuiModule> uptr);

        void remove(const std::string &name);

        void remove(std::unique_ptr<GuiModule> uptr);

        void render_menu();

        void render_gui();

        std::unordered_map<std::string, std::unique_ptr<GuiModule>> gui_modules;
        bool active = false;
    };
}

#endif //GUIMODULES_H
