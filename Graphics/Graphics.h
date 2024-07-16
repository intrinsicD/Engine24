//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_GRAPHICS_H
#define ENGINE24_GRAPHICS_H

#include <unordered_map>
#include <vector>
#include <string>
#include "MatVec.h"

struct GLFWwindow;

namespace Bcg {

    struct BufferContainer : public std::unordered_map<std::string, unsigned int> {
        using std::unordered_map<std::string, unsigned int>::unordered_map;
    };

    struct ShaderContainer : public std::unordered_map<std::string, unsigned int> {
        using std::unordered_map<std::string, unsigned int>::unordered_map;
    };

    struct ProgramContainer : public std::unordered_map<std::string, unsigned int> {
        using std::unordered_map<std::string, unsigned int>::unordered_map;
    };

    struct BatchedBuffer {
        unsigned int id = -1;
        unsigned int usage = -1;
        unsigned int target = -1;
        unsigned int type = -1;

        size_t total_size_bytes() const {
            size_t size_in_bytes = 0;
            for (const auto &item: layout) {
                size_in_bytes += item.size_in_bytes;
            }
            return size_in_bytes;
        }

        struct Layout {
            const char *name;
            unsigned int size_in_bytes = 0;
            unsigned int dims = 0;          //3
            unsigned int size = 0;          //sizeof(float)
            unsigned int normalized = 0;
            unsigned int offset = 0;
            const void *data;

            unsigned int stride() const {
                return size * dims;
            }
        };

        Layout &get_or_add(const char *name) {
            auto iter = std::find_if(layout.begin(), layout.end(), [name](auto &&item) {
                return item.name == name;
            });
            if (iter == layout.end()) {
                Layout item;
                item.name = name;
                item.offset = total_size_bytes();
                layout.push_back(item);
                return layout.back();
            }
            return *iter;
        }

        std::vector<Layout> layout;
    };

    class Graphics {
    public:
        static bool init();

        static bool should_close();

        static void poll_events();

        static void set_window_title(const char *title);

        static void set_clear_color(const float *color);

        static void clear_framebuffer();

        static void start_gui();

        static void render_menu();

        static void render_gui();

        static void end_gui();

        static void swap_buffers();

        static Vector<int, 4> get_viewport();

        static bool read_depth_buffer(int x, int y, float &z);

        //--------------------------------------------------------------------------------------------------------------

        static void setup_batched_buffer(BatchedBuffer &batched_buffer);

        static size_t remove_buffer(const std::string &name);

        static size_t remove_buffer(unsigned int id);

        static size_t buffer_size(unsigned int id, unsigned int target);

        static unsigned int get_or_add_buffer(const std::string &name);

        static void
        upload(unsigned int id, unsigned int target, const void *data, size_t size_bytes, size_t offset = 0);

        static void upload_vbo(unsigned int id, const void *data, size_t size_bytes, size_t offset = 0);

        static void upload_ebo(unsigned int id, const void *data, size_t size_bytes, size_t offset = 0);

        static void upload_ssbo(unsigned int id, const void *data, size_t size_bytes, size_t offset = 0);

        static void upload_ubo(unsigned int id, const void *data, size_t size_bytes, size_t offset = 0);

        static void render_gui(const BufferContainer &buffers);

        //--------------------------------------------------------------------------------------------------------------
    };
}

#endif //ENGINE24_GRAPHICS_H
