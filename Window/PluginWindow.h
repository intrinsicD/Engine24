//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_PLUGINWINDOW_H
#define ENGINE24_PLUGINWINDOW_H

#include "Plugin.h"

namespace Bcg {
    struct WindowInit {
        int width, height;
        const char *title;
    };

    struct WindowInfo {
        void *handle;
        const char *title;
    };

    class PluginWindow : public Plugin {
    public:
        PluginWindow() : Plugin("Window") {}

        static bool init(WindowInit &init);

        static void *create(int width, int height, const char *title);

        static void make_current(void *handle);

        static void *get_current();

        static bool destroy(void *handle);

        static const char *get_title(void *handle);

        static void set_title(void *handle, const char *title);

        static void poll_events();

        static void swap_buffers(void *handle);

        static void swap_buffers();

        static bool should_close(void *handle);

        static void shutdown();

        static float get_dpi_scaling(void *handle);

        static void get_width_and_height(void *handle, int &width, int &height);
    };
}
#endif //ENGINE24_PLUGINWINDOW_H
