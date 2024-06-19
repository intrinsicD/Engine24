//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_ENGINE_H
#define ENGINE24_ENGINE_H

#include "entt/entt.hpp"
#include "Plugin.h"

namespace Bcg {
    struct PluginRenderer;

    struct Engine {
        Engine();

        static Engine *Instance();

        //Main way to have access to the engines state
        static entt::registry &State();

        //Main way to have access to the engines state context
        static entt::registry::context &Context();

        static entt::dispatcher &Dispatcher();

        static void ExecuteCmdBuffer();

        static void ExecuteRenderCmdBuffer();

        entt::registry state;
        entt::dispatcher dispatcher;
        std::vector<std::function<void()>> command_buffer;
        std::vector<std::function<void()>> render_command_buffer;
        std::vector<std::unique_ptr<Plugin>> plugins;
        PluginRenderer *renderer = nullptr;
    };
}

#endif //ENGINE24_ENGINE_H
