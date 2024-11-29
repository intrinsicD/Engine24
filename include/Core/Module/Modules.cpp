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
}