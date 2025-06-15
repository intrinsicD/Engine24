//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_APPLICATION_H
#define ENGINE24_APPLICATION_H

#include "Engine.h"
#include "Window.h"
#include "Renderer.h"

namespace Bcg {
    class Application {
    public:
        Application();

        void init(int width, int height, const char *title);

        void run();

        void cleanup();

        Engine engine;
        std::unique_ptr<Window> window;
        std::unique_ptr<Renderer> renderer;
    };
}

#endif //ENGINE24_APPLICATION_H
