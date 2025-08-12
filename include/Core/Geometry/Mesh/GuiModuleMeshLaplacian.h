//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_GUIMODULEMESHLAPLACIAN_H
#define ENGINE24_GUIMODULEMESHLAPLACIAN_H

#include "GuiModule.h"
#include "SurfaceMeshLaplacianOperator.h"

namespace Bcg {
    class GuiModuleMeshLaplacian : public GuiModule {
    public:
        GuiModuleMeshLaplacian(entt::registry &registry);

        ~GuiModuleMeshLaplacian() override = default;

        void render_menu() override;

        void render_gui() override;

    private:
        // Recursively draws an entity and all its children as a tree.
        void render_gui(entt::entity entity_id);

        entt::registry &m_registry; // Provides access to the entity registry and components
        bool m_is_window_open = false; // Controls the visibility of the hierarchy window
    };
}

#endif //ENGINE24_GUIMODULEMESHLAPLACIAN_H
