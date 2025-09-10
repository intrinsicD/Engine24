#pragma once

#include "entt/entity/registry.hpp"

namespace Bcg {
    template<typename T>
    T &Require(entt::entity entity_id, entt::registry &registry) {
        if (!registry.all_of<T>(entity_id)) {
            return registry.emplace<T>(entity_id);
        }
        return registry.get<T>(entity_id);
    }
}