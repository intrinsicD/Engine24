//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINAABB_H
#define ENGINE24_PLUGINAABB_H

#include "Module.h"


namespace Bcg {
    class ModuleAABB : public Module {
    public:
        explicit ModuleAABB();

        ~ModuleAABB() override = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;
        
        void render() override;

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void center_and_scale_by_aabb(entt::entity entity_id, const std::string &property_name);
    };
}

#endif //ENGINE24_PLUGINAABB_H
