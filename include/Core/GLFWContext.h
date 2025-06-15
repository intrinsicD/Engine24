//
// Created by alex on 6/15/25.
//

#ifndef GLFWCONTEXT_H
#define GLFWCONTEXT_H

namespace Bcg {
    class GLFWContext {
    public:
        bool init();

        void shutdown();

    private:
        static void glfw_error_callback(int error, const char *description);
    };
}

#endif //GLFWCONTEXT_H
