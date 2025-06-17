//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_HIERARCHY_H
#define ENGINE24_HIERARCHY_H

#include "entt/fwd.hpp"

namespace Bcg {
    class Hierarchy{
    public:
        explicit Hierarchy(entt::registry &registry);

        void set_parent(entt::entity child_id, entt::entity parent_id);

        void destroy_entity(entt::entity entity_id);

    protected:
        entt::registry &m_registry;
    };
}

#endif //ENGINE24_HIERARCHY_H
