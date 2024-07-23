//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_APPLICATION_H
#define ENGINE24_APPLICATION_H

#include "Engine.h"
#include "bgfx/bgfx.h"

namespace Bcg {
    class Application {
    public:
        Application();

        void init(int width, int height, const char *title);

        void run();

        void cleanup();

        Engine engine;
        bgfx::Init bgfx;
    };
}

#endif //ENGINE24_APPLICATION_H
