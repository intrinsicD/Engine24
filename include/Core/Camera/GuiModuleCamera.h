//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_GUIMODULECAMERA_H
#define ENGINE24_GUIMODULECAMERA_H

#include "GuiModule.h"

namespace Bcg{
    class GuiModuleCamera : public GuiModule {
    public:
        explicit GuiModuleCamera();

        ~GuiModuleCamera() override = default;

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;
    };
}

#endif //ENGINE24_GUIMODULECAMERA_H
