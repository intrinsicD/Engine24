//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_ENGINE_H
#define ENGINE24_ENGINE_H

#include "entt/entt.hpp"
#include "Command.h"

namespace Bcg {
    struct Engine {
        Engine();

        static Engine *Instance();

        //Main way to have access to the engines state
        static entt::registry &State();

        //Main way to have access to the engines state context
        static entt::registry::context &Context();

        static entt::dispatcher &Dispatcher();

        static void ExecuteCmdBuffer();

        entt::registry state;
        entt::dispatcher dispatcher;
    };
}

#endif //ENGINE24_ENGINE_H
