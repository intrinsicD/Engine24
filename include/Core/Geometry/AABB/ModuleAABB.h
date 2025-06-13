//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINAABB_H
#define ENGINE24_PLUGINAABB_H

#include "ComponentModule.h"
#include "AABBUtils.h"
#include "MemoryPool.h"


namespace Bcg {
    using AABBHandle = PoolHandle<AABB<float>>;
    using AABBPool = Pool<AABB<float>>;

    class ModuleAABB : public Module {
    public:
        explicit ModuleAABB();

        ~ModuleAABB() override = default;

        void activate() override;

        void deactivate() override;

        // Creation and management --------------------------------------------------------------------------------------

        static AABBHandle make_handle(const AABB<float> &object);

        static AABBHandle create(entt::entity entity_id, const AABB<float> &object);

        static AABBHandle add(entt::entity entity_id, AABBHandle h_object);

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static AABBHandle get(entt::entity entity_id);

        // Processing ---------------------------------------------------------------------------------------------------

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void center_and_scale_by_aabb(entt::entity entity_id, const std::string &property_name);

        // Gui stuff ---------------------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;

        static void show_gui(const PoolHandle<AABB<float>> &h_aabb);

        static void show_gui(const AABB<float> &aabb);

        static void show_gui(entt::entity entity_id);

        // Events ---------------------------------------------------------------------------------------------------
    };
}

#endif //ENGINE24_PLUGINAABB_H
