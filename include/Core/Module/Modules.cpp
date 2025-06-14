//
// Created by alex on 26.11.24.
//

#include "Modules.h"
#include <unordered_map>

namespace Bcg{
    static std::unordered_map<std::string, std::unique_ptr<Module>> modules;

    void Modules::activate(){
        for(auto &module : modules){
            module.second->activate();
        }
    }

    void Modules::deactivate(){
        for(auto &module : modules){
            module.second->deactivate();
        }
    }

    void Modules::add(std::unique_ptr<Module> uptr){
        modules[uptr->get_name()] = std::move(uptr);
    }

    void Modules::remove(const std::string &name){
        modules.erase(name);
    }

    void Modules::remove(std::unique_ptr<Module> uptr){
        modules.erase(uptr->get_name());
    }

    void Modules::begin_frame(){
        for(auto &module : modules){
            module.second->begin_frame();
        }
    }

    void Modules::update(){
        for(auto &module : modules){
            module.second->update();
        }
    }

    void Modules::fixed_update(double fixed_time_step) {

    }

    void Modules::variable_update(double delta_time, double alpha) {

    }

    void Modules::render(){
        for(auto &module : modules){
            module.second->render();
        }
    }

    void Modules::render_menu() {
        for(auto &module : modules){
            module.second->render_menu();
        }
    }

    void Modules::render_gui() {
        for(auto &module : modules){
            module.second->render_gui();
        }
    }


    void Modules::end_frame(){
        for(auto &module : modules){
            module.second->end_frame();
        }
    }


}