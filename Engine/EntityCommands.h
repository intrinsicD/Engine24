//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_ENTITYCOMMANDS_H
#define ENGINE24_ENTITYCOMMANDS_H

#include "Command.h"
#include "Engine.h"
#include "Logger.h"
#include "EventsEntity.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands::Entity {
    template<typename Component>
    struct Add : public Task {
        Add(entt::entity entity, std::string component_name = "") :
                Task("Add " + component_name,
                     [entity, this]() {
                         if (!Engine::valid(entity)) {
                             return;
                         }
                         if (!Engine::has<Component>(entity)) {
                             Engine::Dispatcher().trigger(Events::Entity::PreAdd<Component>{entity});

                             Engine::State().emplace<Component>(entity);

                             Engine::Dispatcher().trigger(Events::Entity::PostAdd<Component>{entity});
                             Log::Info(name + ". Done.");
                         } else {
                             Log::Warn(name + ". Failed, already exists.");
                         }
                     }) {}

        Add(entt::entity entity, Component &component, std::string component_name = "") :
                Task("Add " + component_name,
                     [entity, component, this]() mutable {
                         if (!Engine::valid(entity)) {
                             return;
                         }
                         Engine::Dispatcher().trigger(Events::Entity::PreAdd<Component>{entity});
                         Engine::State().emplace_or_replace<Component>(entity, component);
                         Engine::Dispatcher().trigger(Events::Entity::PostAdd<Component>{entity});
                         Log::Info(name + ". Done.");
                     }) {}

        Add(entt::entity entity, Component &&component, std::string component_name = "") :
                Task("Add " + component_name,
                     [entity, component = std::move(component), this]() mutable {
                         if (!Engine::valid(entity)) {
                             return;
                         }
                         Engine::Dispatcher().trigger(Events::Entity::PreAdd<Component>{entity});
                         Engine::State().emplace_or_replace<Component>(entity, std::move(component));
                         Engine::Dispatcher().trigger(Events::Entity::PostAdd<Component>{entity});
                         Log::Info(name + ". Done.");
                     }) {}

        ~Add() override = default;

        entt::entity entity;
    };

    template<typename Component>
    struct Remove : public AbstractCommand {
        Remove(entt::entity entity, std::string component_name = "") : AbstractCommand("Remove " + component_name),
                                                                       entity(entity) {}

        ~Remove() override = default;

        void execute() const override {
            if (!Engine::valid(entity)) {
                return;
            }
            if (Engine::has<Component>(entity)) {
                Engine::Dispatcher().trigger(Events::Entity::PreRemove<Component>{entity});

                Engine::State().remove<Component>(entity);

                Engine::Dispatcher().trigger(Events::Entity::PostRemove<Component>{entity});
                Log::Info(name + ". Done.");
            } else {
                Log::Warn(name + ". Failed, does not exists.");
            }
        }

        entt::entity entity;
    };
}

#endif //ENGINE24_ENTITYCOMMANDS_H
