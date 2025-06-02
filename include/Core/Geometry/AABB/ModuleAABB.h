//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINAABB_H
#define ENGINE24_PLUGINAABB_H

#include "ComponentModule.h"
#include "AABB.h"


namespace Bcg {
    using AABBHandle = PoolHandle<AABB>;
    using AABBPool = Pool<AABB>;

    class ModuleAABB : public Module {
    public:
        explicit ModuleAABB();

        ~ModuleAABB() override = default;

        void activate() override;

        void deactivate() override;

        static AABBHandle make_handle(const AABB &object);

        static AABBHandle create(entt::entity entity_id, const AABB &object);

        static AABBHandle add(entt::entity entity_id, AABBHandle h_object);

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static AABBHandle get(entt::entity entity_id);

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void render() override;

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void center_and_scale_by_aabb(entt::entity entity_id, const std::string &property_name);
    };
}

#endif //ENGINE24_PLUGINAABB_H
