//
// Created by alex on 11/24/24.
//

#ifndef RENDERMODULE_H
#define RENDERMODULE_H

#include "Module/Module.h"

namespace Bcg {
    class RenderModule : public Module {
    public:
        ~RenderModule() override = default;

        virtual void begin_frame() = 0;

        virtual void render() = 0;

        virtual void end_frame() = 0;
    };
}

#endif //RENDERMODULE_H
