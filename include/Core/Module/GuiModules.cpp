//
// Created by alex on 25.11.24.
//

#include <unordered_map>
#include "GuiModules.h"

namespace Bcg {
    static std::unordered_map<std::string, std::unique_ptr<GuiModule>> gui_modules;

    void GuiModules::activate() {
        if(active){
            Log::Info("Already active GuiModules");
            return;
        }else{
            active = true;
            Log::Info("Activate GuiModules");
            for(auto &module: gui_modules){
                module.second->activate();
            }
        }
    }

    void GuiModules::deactivate() {
        if(!active){
            Log::Info("Already deactivated GuiModules");
            return;
        }else{
            Log::Info("Deactivate GuiModules");
            for(auto &module: gui_modules){
                module.second->deactivate();
            }
            active = false;
        }
    }

    void GuiModules::add(std::unique_ptr<GuiModule> uptr) {
        gui_modules[uptr->get_name()] = std::forward<std::unique_ptr<GuiModule>>(uptr);
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