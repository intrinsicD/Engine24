#include <vector>

#include "GLFW/glfw3.h"
#include "entt/entt.hpp"
#include "Engine.h"
#include "PluginGlfwWindow.h"

// The MAIN function, from here we start the application and run the game loop
int main() {
    Bcg::Engine engine;

    auto window_plugin = std::make_unique<Bcg::PluginGlfwWindow>();
    window_plugin->init();
    auto &plugins = Bcg::Engine::Instance()->plugins;
    plugins.emplace_back(std::move(window_plugin));

    for (auto &plugin: plugins) {
        plugin->activate();
    }

    // Game loop
    while (!glfwWindowShouldClose(Bcg::Engine::Instance()->window)) {
        // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();

        {
            for (auto &plugin: plugins) {
                plugin->update();
            }
        }

        Bcg::Engine::ExecuteCmdBuffer();

        window_plugin->clear_framebuffer();
        {
            for (auto &plugin: plugins) {
                plugin->render();
            }
        }

        Bcg::Engine::ExecuteRenderCmdBuffer();

        {
            window_plugin->start_gui();
            for (auto &plugin: plugins) {
                plugin->render_menu();
            }

            for (auto &plugin: plugins) {
                plugin->render_gui();
            }
            window_plugin->end_gui();
        }

        // Swap the screen buffers
        glfwSwapBuffers(Bcg::Engine::Instance()->window);
    }

    for (auto &plugin: plugins) {
        plugin->deactivate();
    }
    // Terminates GLFW, clearing any resources allocated by GLFW.
    glfwDestroyWindow(Bcg::Engine::Instance()->window);
    glfwTerminate();
    return 0;
}