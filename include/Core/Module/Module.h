//
// Created by alex on 11/24/24.
//

#ifndef MODULE_H
#define MODULE_H

#include <string>

namespace Bcg {
    class Module {
    public:
        Module(const std::string &name) : name(name) {}

        virtual ~Module() = default;

        virtual void activate() = 0; //registers callbacks to events

        virtual void deactivate() = 0; //unregisters callbacks to events

        std::string name;
    };
}

#endif //MODULE_H
