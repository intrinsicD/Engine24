//
// Created by alex on 6/4/25.
//

#ifndef MODULESHADERSLANG_H
#define MODULESHADERSLANG_H

#include "Module.h"

namespace Bcg {
    class ModuleShaderSlang : public Module {
    public:
        ModuleShaderSlang() : Module("ModuleShaderSlang") {

        }

        void activate() override;

        void deactivate() override;

        void update() override;

        void render_menu() override;

        void render_gui() override;

        //static methods for shader compilation and management
    };
}

#endif //MODULESHADERSLANG_H
