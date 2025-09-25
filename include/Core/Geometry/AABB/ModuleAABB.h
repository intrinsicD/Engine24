//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINAABB_H
#define ENGINE24_PLUGINAABB_H

#include "ComponentModule.h"
#include "AABBUtils.h"
#include "MemoryPool.h"


namespace Bcg {
    class ModuleAABB : public Module {
    public:
        explicit ModuleAABB();

        ~ModuleAABB() override = default;

        void activate() override;

        void deactivate() override;

        // Processing ---------------------------------------------------------------------------------------------------

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        // Gui stuff ---------------------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;

        static void show_gui(const char *label, const AABB<float> &aabb);

        static void show_gui(entt::entity entity_id);

        // Events ---------------------------------------------------------------------------------------------------
    };
}

#endif //ENGINE24_PLUGINAABB_H
