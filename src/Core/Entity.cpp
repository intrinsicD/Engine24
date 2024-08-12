//
// Created by alex on 12.08.24.
//

#include "Entity.h"
#include "Engine.h"
#include "Logger.h"

namespace Bcg {
    Entity::Entity(entt::entity entity_id) : entity_id(entity_id) {}

    Entity::operator bool() const {
        bool is_valid = Engine::valid(entity_id);
        if (!is_valid) Log::Warn("Entity is not valid: {}", entity_id);
        return is_valid;
    }

    Entity::operator entt::entity() const { return entity_id; }

    entt::entity Entity::id() const { return entity_id; }

    std::string Entity::to_string() const {
        return fmt::format("Entity: {}", entity_id);
    }

}