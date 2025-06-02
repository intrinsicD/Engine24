//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_GUIMODULETRANSFORM_H
#define ENGINE24_GUIMODULETRANSFORM_H

#include "GuiModule.h"

namespace Bcg{
    class GuiModuleTransform : public GuiModule {
    public:
        GuiModuleTransform();

        ~GuiModuleTransform() override = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;
    };
}

#endif //ENGINE24_GUIMODULETRANSFORM_H
