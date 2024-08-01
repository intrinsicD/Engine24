//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_SHADER_H
#define ENGINE24_SHADER_H

#include <string>

namespace Bcg {
    struct Shader {
        unsigned int id = -1;
        unsigned int type;
        std::string source;
        std::string filepath;

        operator bool() const {
            return id != -1;
        }

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
}

#endif //ENGINE24_SHADER_H
