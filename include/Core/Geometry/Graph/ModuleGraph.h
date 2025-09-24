#pragma once

#include "ComponentModule.h"
#include "ComponentHandle.h"
#include "Graph.h"
#include "StringTraitsMesh.h"
#include "Events/EventsCallbacks.h"

namespace Bcg {
    class ModuleGraph : public Module {
    public:
        ModuleGraph();

        ~ModuleGraph() override = default;

        void activate() override;

        void deactivate() override;

        // Creation and management -------------------------------------------------------------------------------------

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static void destroy_entity(entt::entity entity_id);


        // Processing --------------------------------------------------------------------------------------------------

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        // Gui stuff ---------------------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;


        static void show_gui(const GraphInterface &pci);

        static void show_gui(entt::entity entity_id);
    };
}
