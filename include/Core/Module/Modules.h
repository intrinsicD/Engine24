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

        void activate();

        void deactivate();

        void add(std::unique_ptr<Module> uptr);

        void remove(const std::string &name);

        void remove(std::unique_ptr<Module> uptr);

        void update();
    };
}
#endif //ENGINE24_MODULES_H
