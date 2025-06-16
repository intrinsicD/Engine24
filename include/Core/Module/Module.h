//
// Created by alex on 11/24/24.
//

#ifndef MODULE_H
#define MODULE_H

#include <string>
#include <utility>
#include "Logger.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class Module {
    public:
        explicit Module(std::string name) : name(std::move(name)), activated(false) {
        }

        virtual ~Module() = default;

        [[nodiscard]] const std::string &get_name() const { return name; }

        [[nodiscard]] bool is_activated() const { return activated; }

        virtual void activate(){} //registers callbacks to events

        virtual void deactivate(){} //unregisters callbacks to events

        // Optional lifecycle hooks
        virtual void begin_frame() {
        }

        virtual void update() {
        }

        virtual void render() {
        }

        virtual void render_menu() {
        }

        virtual void render_gui() {
        }

        virtual void end_frame() {
        }

    protected:
        bool base_activate() {
            if (!activated) {
                activated = true;
                Log::Info("Activate {}", name);
                return true;
            } else {
                Log::Warn("Already active {}", name);
                return false;
            }
        }

        bool base_deactivate() {
            if (activated) {
                activated = false;
                Log::Info("Deactivate {}", name);
                return true;
            } else {
                Log::Warn("Already deactivated {}", name);
                return false;
            }
        }

        std::string name;
        bool activated;
    };
}

#endif //MODULE_H
