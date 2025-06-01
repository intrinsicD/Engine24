//
// Created by alex on 25.11.24.
//

#ifndef AABBGUIMODULE_H
#define AABBGUIMODULE_H

#include "GuiModule.h"
#include "AABBPool.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class GuiModuleAABB : public GuiModule {
    public:
        GuiModuleAABB() : GuiModule("AABBGuiModule") {};

        ~GuiModuleAABB() override = default;

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        static void show(const PoolHandle<AABB> &h_aabb);

        static void show(const AABB &aabb);

        static void show(Pool<AABB> &pool);

        static void show(entt::entity entity_id);
    };
}

#endif //AABBGUIMODULE_H
