#pragma once

#include "GuiModule.h"

namespace Bcg {
    class GuiModuleGaussianMixture : public GuiModule {
    public:
        explicit GuiModuleGaussianMixture(entt::registry &registry);

        ~GuiModuleGaussianMixture() override = default;

        void render_menu() override;

        void render_gui() override;

    private:
        // Recursively draws an entity and all its children as a tree.
        void render_gui(entt::entity entity_id);

        entt::registry &m_registry; // Provides access to the entity registry and components
        bool m_is_window_open = false; // Controls the visibility of the hierarchy window
    };
}
