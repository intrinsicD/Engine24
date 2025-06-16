//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_GUIMODULERENDERERSETTINGS_H
#define ENGINE24_GUIMODULERENDERERSETTINGS_H

#include "Renderer.h"
#include "GuiModule.h"

namespace Bcg{
    class GuiModuleRendererSettings : public GuiModule{
    public:
        GuiModuleRendererSettings(Renderer &renderer);

        ~GuiModuleRendererSettings() = default;

        void render_menu() override;

        void render_gui() override;

    private:
        Renderer &m_renderer;
        bool m_is_window_open = false;
    };
}

#endif //ENGINE24_GUIMODULERENDERERSETTINGS_H
