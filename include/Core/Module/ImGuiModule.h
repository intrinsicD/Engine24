//
// Created by alex on 25.11.24.
//

#ifndef IMGUIMODULE_H
#define IMGUIMODULE_H

#include "GuiModule.h"

namespace Bcg {
    class ImGuiModule : public GuiModule {
    public:
        ~ImGuiModule() override = default;

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;
    };
}

#endif //IMGUIMODULE_H
