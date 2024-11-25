//
// Created by alex on 25.11.24.
//

#include "GuiModules.h"
#include "Logger.h"

namespace Bcg {
    void GuiModules::activate() {
        active = true;
        Log::Info("GuiModules activated");
    }

    void GuiModules::deactivate() {
        active = false;
        Log::Info("GuiModules deactivated");
    }

    void GuiModules::add(const std::string &name, std::unique_ptr<GuiModule> uptr) {
        gui_modules[name] = std::forward<std::unique_ptr<GuiModule>>(uptr);
    }

    void GuiModules::remove(const std::string &name) {
        gui_modules.erase(name);
    }

    void GuiModules::remove(std::unique_ptr<GuiModule> uptr) {
        for (auto it = gui_modules.begin(); it != gui_modules.end();) {
            if (it->second.get() == uptr.get()) {
                it = gui_modules.erase(it);
            } else {
                ++it;
            }
        }
    }

    void GuiModules::render_menu() {
        if (!active) return;
        for (auto &module: gui_modules) {
            module.second->render_menu();
        }
    }

    void GuiModules::render_gui() {
        if (!active) return;
        for (auto &module: gui_modules) {
            module.second->render_gui();
        }
    }
}