//
// Created by alex on 19.06.24.
//

#include "Engine.h"
#include "entt/entt.hpp"
#include "CommandDoubleBuffer.h"

namespace Bcg {
    Engine::Engine() {
        entt::locator<Bcg::Engine *>::emplace<Bcg::Engine *>(this);
        state.ctx().emplace<DoubleCommandBuffer>();
        assert(Instance() == this);
    }

    bool Engine::valid(entt::entity entity) {
        return State().valid(entity);
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
        auto &double_cmd_buffer = Engine::Context().get<DoubleCommandBuffer>();
        double_cmd_buffer.current().execute();
        double_cmd_buffer.current().clear();
        double_cmd_buffer.swap_buffers();
        Engine::Dispatcher().update();
        Engine::Dispatcher().clear();
    }
}