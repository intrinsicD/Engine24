//
// Created by alex on 11/24/24.
//

#ifndef MODULE_H
#define MODULE_H

#include <string>
#include "Logger.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class Module {
    public:
        explicit Module(const std::string &name) : name(name), activated(false) {
        }

        virtual ~Module() = default;

        const std::string &get_name() const { return name; }

        bool is_activated() const { return activated; }

        virtual void activate() = 0; //registers callbacks to events

        virtual void deactivate() = 0; //unregisters callbacks to events

        // Optional lifecycle hooks
        virtual void begin_frame() {
        }

        virtual void update() {
        }

        virtual void end_frame() {
        }

        virtual void render() {
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
