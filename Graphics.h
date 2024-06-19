//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_GRAPHICS_H
#define ENGINE24_GRAPHICS_H

#include <vector>

namespace Bcg {
    class Graphics {
    public:
        bool init();

        bool should_close() const;

        void poll_events() const;

        void set_clear_color(const float *color);

        void clear_framebuffer() const;

        void start_gui() const;

        void render_menu() const;

        void render_gui() const;

        void end_gui() const;

        void swap_buffers() const;

        unsigned int load_shader(const char *path, unsigned int type);

        unsigned int compile_shader(const char *source, unsigned int type);

        unsigned int load_program(const char *v_path,
                                  const char *f_path,
                                  const char *g_path = "",
                                  const char *tc_path = "",
                                  const char *te_path = "");

        unsigned int load_compute(const char *path);
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
    };

    class Program {
    public:
        Program();

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
        std::vector<Shader> shaders;
    };
}

#endif //ENGINE24_GRAPHICS_H
