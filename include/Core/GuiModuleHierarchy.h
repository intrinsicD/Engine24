//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_GUIMODULEHIERARCHY_H
#define ENGINE24_GUIMODULEHIERARCHY_H

#include "GuiModule.h"
#include "EntitySelection.h"

namespace Bcg {
    class GuiModuleHierarchy : public GuiModule {
    public:
        GuiModuleHierarchy(entt::registry &registry, EntitySelection &entity_selection);

        ~GuiModuleHierarchy() override = default;

        void render_menu() override;

        void render_gui() override;

    private:
        // Recursively draws an entity and all its children as a tree.
        void draw_entity_node(entt::entity entity_id);

        entt::registry &m_registry; // Provides access to the entity registry and components
        EntitySelection &m_entity_selection; // Provides access to the currently selected entity
        bool m_is_window_open = false; // Controls the visibility of the hierarchy window
    };
}

#endif //ENGINE24_GUIMODULEHIERARCHY_H
