//
// Created by alex on 19.06.24.
//

#include "Engine.h"

namespace Bcg {
    Engine::Engine() {
        entt::locator<Bcg::Engine *>::emplace<Bcg::Engine *>(this);
        assert(Instance() == this);
    }

    Engine *Engine::Instance() {
        return entt::locator<Engine *>::value();
    }

    //Main way to have access to the engines state
    entt::registry &Engine::State() {
        return Instance()->state;
    }

    //Main way to have access to the engines state context
    entt::registry::context &Engine::Context() {
        return State().ctx();
    }

    entt::dispatcher &Engine::Dispatcher() {
        return Instance()->dispatcher;
    }

    void Engine::ExecuteCmdBuffer() {
        Dispatcher().update();
        Dispatcher().clear();
    }
}