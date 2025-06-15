//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MODULES_H
#define ENGINE24_MODULES_H

#include <memory>
#include "Module.h"

namespace Bcg {
    class Modules {
    public:
        Modules() = default;

        ~Modules() = default;

        void add(std::unique_ptr<Module> uptr);

        void remove(const std::string &name);

        void remove(std::unique_ptr<Module> uptr);

        void activate();

        void deactivate();

        void begin_frame();

        void update();

        void fixed_update(double fixed_time_step);

        void variable_update(double delta_time, double alpha);

        void render();

        void render_menu();

        void render_gui();

        void end_frame();
    };
}
#endif //ENGINE24_MODULES_H
