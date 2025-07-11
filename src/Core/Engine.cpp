//
// Created by alex on 19.06.24.
//

#include "Engine.h"
#include "entt/entt.hpp"
#include "CommandDoubleBuffer.h"

namespace Bcg {
    Engine::Engine() {
        entt::locator<Engine *>::emplace<Engine *>(this);
        state.ctx().emplace<DoubleCommandBuffer>();
        assert(Instance() == this);
    }

    bool Engine::valid(entt::entity entity) {
        return entity != entt::null && State().valid(entity);
    }

    Engine *Engine::Instance() {
        return entt::locator<Engine *>::value();
    }

    //Main way to have access to the engine's state
    entt::registry &Engine::State() {
        return Instance()->state;
    }

    //Main way to have access to the engine's state context
    entt::registry::context &Engine::Context() {
        return State().ctx();
    }

    entt::dispatcher &Engine::Dispatcher() {
        return Instance()->dispatcher;
    }

    void Engine::handle_command_double_buffer() {
        auto &double_cmd_buffer = Context().get<DoubleCommandBuffer>();
        double_cmd_buffer.handle();
    }

    void Engine::handle_buffered_events() {
        Dispatcher().update();
        Dispatcher().clear();
    }
}