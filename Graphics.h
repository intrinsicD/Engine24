//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_GRAPHICS_H
#define ENGINE24_GRAPHICS_H

#include <vector>

namespace Bcg {
    class Graphics {
    public:
        static bool init();

        static bool should_close();

        static void poll_events();

        static void set_clear_color(const float *color);

        static void clear_framebuffer();

        static void start_gui();

        static void render_menu();

        static void render_gui();

        static void end_gui();

        static void swap_buffers();
    };

    class Shader {
    public:
        Shader(unsigned int type);

        static Shader VertexShader();

        static Shader FragmentShader();

        static Shader GeometryShader();

        static Shader TessContrlShader();

        static Shader TessEvalShader();

        static Shader ComputeShader();

        bool load(const char *path);

        bool compile();

        operator bool() const { return id != -1; }

        unsigned int id = -1;
        unsigned int type;
        const char *source;
        const char *path;
    };

    class Program {
    public:
        Program(const char *name);

        bool load(const char *v_path, const char *f_path, const char *g_path, const char *tc_path, const char *te_path);

        bool load(const char *c_path);

        bool link(const Shader &vs, const Shader &fs,
                  const Shader &gs = Shader::GeometryShader(),
                  const Shader &tcs = Shader::TessContrlShader(),
                  const Shader &tes = Shader::TessEvalShader());

        bool link(const Shader &cs);

        bool link();

        operator bool() const { return id != -1; }

        unsigned int id = -1;
        const char *name;
        std::vector<Shader> shaders;
    };

    class Buffer {
    public:
        Buffer(const char *name);

        void create();

        void destroy();

        void bind() const;

        void unbind() const;

        void set_data(const void *data, unsigned int size_bytes);

        void set_subdata(const void *data, unsigned int offset_bytes, unsigned int total_bytes);

        const char *name;
        unsigned int id = -1;
        unsigned int size_bytes;
    };
}

#endif //ENGINE24_GRAPHICS_H
