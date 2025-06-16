//
// Created by alex on 6/15/25.
//

#ifndef PLATFORM_H
#define PLATFORM_H

namespace Bcg {
    class Platform {
    public:
        Platform();

        ~Platform();

        static bool is_initialized();

    private:
        static void glfw_error_callback(int error, const char *description);
    };
}

#endif //PLATFORM_H
