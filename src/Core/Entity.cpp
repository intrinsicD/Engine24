//
// Created by alex on 12.08.24.
//

#include "Entity.h"
#include "Engine.h"
#include "Logger.h"

namespace Bcg {
    Entity::Entity() : entity_id(entt::null) {}

    Entity::Entity(entt::entity entity_id) : entity_id(entity_id) {}

    Entity::operator bool() const {
        return is_valid();
    }

    bool Entity::is_valid() const {
        return Engine::valid(entity_id);
    }

    Entity::operator entt::entity() const { return entity_id; }

    Entity &Entity::create() {
        if (!is_valid()) {
            entity_id = Engine::State().create();
        }
        return *this;
    }

    Entity &Entity::destroy() {
        if (is_valid()) {
            Engine::State().destroy(entity_id);
            entity_id = entt::null;
        }
        return *this;
    }

    entt::entity Entity::id() const { return entity_id; }

    std::string Entity::to_string() const {
        return fmt::format("Entity: {}", entity_id);
    }

}