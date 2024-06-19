#include <vector>

#include "entt/entt.hpp"
#include "Engine.h"
#include "Input.h"
#include "Graphics.h"
#include "PluginMesh.h"

int main() {
    Bcg::Engine engine;
    Bcg::Graphics graphics;
    if(!graphics.init()){
        return -1;
    }

    auto &plugins = Bcg::Engine::Instance()->plugins;
    plugins.emplace_back(std::make_unique<Bcg::Input>());

    auto mesh_plugin = std::make_unique<Bcg::PluginMesh>();
    plugins.emplace_back(std::move(mesh_plugin));


    for (auto &plugin: plugins) {
        plugin->activate();
    }

    Bcg::Engine::Dispatcher().update();

    // Game loop
    while (!graphics.should_close()) {
        for (auto &plugin: plugins) {
            plugin->begin_frame();
        }
        // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
        graphics.poll_events();

        {
            for (auto &plugin: plugins) {
                plugin->update();
            }
        }

        Bcg::Engine::ExecuteCmdBuffer();

        graphics.clear_framebuffer();
        {
            for (auto &plugin: plugins) {
                plugin->render();
            }
        }

        Bcg::Engine::ExecuteRenderCmdBuffer();

        {
            graphics.start_gui();
            for (auto &plugin: plugins) {
                plugin->render_menu();
            }

            for (auto &plugin: plugins) {
                plugin->render_gui();
            }
            graphics.render_menu();
            graphics.render_gui();
            graphics.end_gui();
        }

        // Swap the screen buffers
        graphics.swap_buffers();
        for (auto &plugin: plugins) {
            plugin->end_frame();
        }
    }

    for (auto &plugin: plugins) {
        plugin->deactivate();
    }
    return 0;
}