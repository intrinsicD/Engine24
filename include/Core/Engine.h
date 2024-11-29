//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_ENGINE_H
#define ENGINE24_ENGINE_H

#include "entt/entity/registry.hpp"
#include "entt/signal/dispatcher.hpp"

namespace Bcg {
    struct Engine {
        Engine();

        static bool valid(entt::entity entity);

        template<typename Component>
        static bool has(entt::entity entity) {
            return State().all_of<Component>(entity);
        }

        template<typename Component>
        static Component &require(entt::entity entity) {
            return State().get_or_emplace<Component>(entity);
        }

        static Engine *Instance();

        //Main way to have access to the engines state
        static entt::registry &State();

        //Main way to have access to the engines state context
        static entt::registry::context &Context();

        static entt::dispatcher &Dispatcher();

        static void handle_command_double_buffer();

        static void handle_buffered_events();

        entt::registry state;
        entt::dispatcher dispatcher;
    };
}

#endif //ENGINE24_ENGINE_H
