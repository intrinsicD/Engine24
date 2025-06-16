//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_GUIMODULETRANSFORMS_H
#define ENGINE24_GUIMODULETRANSFORMS_H

#include "GuiModule.h"
#include "entt/fwd.hpp"

namespace Bcg{
    class Renderer;
    class EntitySelection;
    class GuiModuleTransforms : public GuiModule{
    public:
        GuiModuleTransforms(entt::registry &registry, Renderer &renderer, EntitySelection &entity_selection);

        ~GuiModuleTransforms() = default;

        void render_menu() override;

        void render_gui() override;

    private:
        // Renders the ImGui inspector panel with text inputs.
        void render_inspector_panel(entt::entity entity_id);

        // Renders the 3D viewport gizmo.
        void render_gizmo(entt::entity entity_id);

        entt::registry &m_registry; // Provides access to the entity registry and components
        Renderer& m_renderer; // Provides camera and registry access
        EntitySelection& m_entity_selection; // Provides access to the currently selected entity

        bool m_is_window_open = false;
    };
}


#endif //ENGINE24_GUIMODULETRANSFORMS_H
