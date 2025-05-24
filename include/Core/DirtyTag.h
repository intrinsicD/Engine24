//
// Created by alex on 5/24/25.
//

#ifndef DIRTYTAG_H
#define DIRTYTAG_H

#include "entt/entity/registry.hpp"

namespace Bcg {
    template<typename Component>
    struct Dirty {

    };

    template<typename Component>
    bool MarkComponentDirty(entt::entity entity_id, entt::registry &registry) {
        if (!registry.valid(entity_id)) { return false; }
        if (!registry.all_of<Component>(entity_id)) { return false; }
        return registry.emplace_or_replace<Dirty<Component>>(entity_id);
    }

    template<typename Component>
    bool RemoveComponentDirty(entt::entity entity_id, entt::registry &registry) {
        if (!registry.valid(entity_id)) { return false; }
        if (!registry.all_of<Component>(entity_id)) { return false; }
        return registry.remove<Dirty<Component>>(entity_id);
    }
}
#endif //DIRTYTAG_H
