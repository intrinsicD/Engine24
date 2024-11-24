//
// Created by alex on 11/24/24.
//

#ifndef MODULE_H
#define MODULE_H

namespace Bcg {
    class Module {
    public:
        virtual ~Module() = default;

        virtual void activate() = 0; //registers callbacks to events

        virtual void deactivate() = 0; //unregisters callbacks to events
    };
}

#endif //MODULE_H
