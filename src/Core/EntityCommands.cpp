//
// Created by alex on 12.08.24.
//

#include "EntityCommands.h"
#include "EventsEntity.h"
#include "Engine.h"

namespace Bcg::Commands::Entity {
    void DestroyEntity::execute() const {
        Engine::Dispatcher().trigger(Events::Entity::Destroy{entity_id});
    }
}