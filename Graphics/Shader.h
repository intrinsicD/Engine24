//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_SHADER_H
#define ENGINE24_SHADER_H

#include <string>

namespace Bcg {
    struct Shader {
        unsigned int id;
        unsigned int type;
        std::string source;

        void create();

        void destroy();

        void load_source(const std::string &source);

        std::string load_file(const std::string &filename);

        void compile();

        bool check_compile_errors();
    };

    struct ComputeShader : public Shader {
        ComputeShader();
    };

    struct VertexShader : public Shader {
        VertexShader();
    };

    struct FragmentShader : public Shader {
        FragmentShader();
    };

    struct GeometryShader : public Shader {
        GeometryShader();
    };

    struct TessControlShader : public Shader {
        TessControlShader();
    };

    struct TessEvalShader : public Shader {
        TessEvalShader();
    };

    struct Program {
        unsigned int id = -1;

        void create();

        void destroy();

        void create_from_source(const std::string &vs, const std::string &fs,
                                const std::string &gs = "", const std::string &tc = "", const std::string &te = "");

        void create_from_files(const std::string &vs, const std::string &fs,
                               const std::string &gs = "", const std::string &tc = "", const std::string &te = "");

        void attach(const Shader &shader);

        void detach(const Shader &shader);

        void link();

        bool check_link_errors();

        void use();

        unsigned int get_uniform_location(const std::string &name);

        unsigned int get_uniform_block_index(const std::string &name);

        void bind_uniform_block(const std::string &name, unsigned int binding_point);

        void set_uniform3fv(const std::string &name, const float *ptr);
    };

    struct ComputeShaderProgram : public Program {
        enum Barrier {
            SHADER_STORAGE_BARRIER_BIT = 0x00002000,
            ATOMIC_COUNTER_BARRIER_BIT = 0x00001000,
            TEXTURE_FETCH_BARRIER_BIT = 0x00000008,
            IMAGE_ACCESS_BARRIER_BIT = 0x00000020,
            COMMAND_BARRIER_BIT = 0x00000040,
            PIXEL_BUFFER_BARRIER_BIT = 0x00000080,
            TEXTURE_UPDATE_BARRIER_BIT = 0x00000100,
            BUFFER_UPDATE_BARRIER_BIT = 0x00000200,
            FRAMEBUFFER_BARRIER_BIT = 0x00000400,
            TRANSFORM_FEEDBACK_BARRIER_BIT = 0x00000800,
            UNIFORM_BARRIER_BIT = 0x00000004
        };

        bool create_from_source(const std::string &cs);

        bool create_from_file(const std::string &cs);

        void dispatch(unsigned int num_groups_x, unsigned int num_groups_y, unsigned int num_groups_z);

        void memory_barrier(unsigned int barriers);
    };


}

#endif //ENGINE24_SHADER_H
