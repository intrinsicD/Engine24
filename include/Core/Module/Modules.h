//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MODULES_H
#define ENGINE24_MODULES_H

#include <memory>
#include <string>
#include <unordered_map>

#include "Module.h"
#include "Utils.h"

namespace Bcg {
    class Modules {
    public:
        Modules() = default;

        // Delete copy constructor and copy assignment operator
        Modules(const Modules&) = delete;
        Modules& operator=(const Modules&) = delete;

        // Default move constructor and move assignment operator
        Modules(Modules&&) noexcept = default;
        Modules& operator=(Modules&&) noexcept = default;

        ~Modules() = default;

        void activate(){
            for(auto &module : modules){
                module.second->activate();
            }
        }

        void deactivate(){
            for(auto &module : modules){
                module.second->deactivate();
            }
        }

        void add(std::unique_ptr<Module> uptr){
            modules[uptr->get_name()] = std::move(uptr);
        }

        void remove(const std::string &name){
            modules.erase(name);
        }

        void remove(std::unique_ptr<Module> uptr){
            modules.erase(uptr->get_name());
        }

        using KeyIterator = KeyIterator<std::string, std::unique_ptr<Module>>;

        KeyIterator begin_keys() const {
            return KeyIterator(modules.cbegin());
        }

        KeyIterator end_keys() const {
            return KeyIterator(modules.cend());
        }

        using ValueIterator = ValueIterator<std::string, std::unique_ptr<Module>>;

        ValueIterator begin_values() const {
            return ValueIterator(modules.cbegin());
        }

        ValueIterator end_values() const {
            return ValueIterator(modules.cend());
        }
    private:
        std::unordered_map<std::string, std::unique_ptr<Module>> modules = {};
    };
}
#endif //ENGINE24_MODULES_H
