//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_ENGINE_H
#define ENGINE24_ENGINE_H

#include "entt/entt.hpp"

struct GLFWwindow;


namespace Bcg {
    struct Plugin;

    struct Engine {
        Engine() {
            entt::locator<Bcg::Engine *>::emplace<Bcg::Engine *>(this);
        }

        static Engine *Instance() {
            return entt::locator<Engine *>::value();
        }

        //Main way to have access to the engines state
        static auto &State() {
            return Instance()->state;
        }

        //Main way to have access to the engines state context
        static auto &Context() {
            return State().ctx();
        }

        static auto &Dispatcher() {
            return Instance()->dispatcher;
        }

        static void ExecuteCmdBuffer() {
            for (auto &cmd: Bcg::Engine::Instance()->command_buffer) {
                cmd();
            }
            Bcg::Engine::Instance()->command_buffer.clear();
        }

        static void ExecuteRenderCmdBuffer() {
            for (auto &cmd: Bcg::Engine::Instance()->render_command_buffer) {
                cmd();
            }
            Bcg::Engine::Instance()->render_command_buffer.clear();
        }

        entt::registry state;
        entt::dispatcher dispatcher;
        GLFWwindow *window = nullptr;
        std::vector<std::function<void()>> command_buffer;
        std::vector<std::function<void()>> render_command_buffer;
        std::vector<std::unique_ptr<Plugin>> plugins;
    };
}

#endif //ENGINE24_ENGINE_H
