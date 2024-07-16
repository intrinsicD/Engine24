//
// Created by alex on 16.07.24.
//


#include "Shader.h"
#include "Logger.h"
#include "glad/gl.h"

namespace Bcg {
    void Shader::create() {
        id = glCreateShader(type);
    }

    void Shader::destroy() {
        glDeleteShader(id);
    }

    void Shader::load_source(const std::string &source) {
        this->source = source;
        const char *src = source.c_str();
        glShaderSource(id, 1, &src, nullptr);
    }

    std::string Shader::load_file(const std::string &filename) {
        FILE *file = fopen(filename.c_str(), "r");
        if (!file) {
            return "";
        }
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char *src = new char[size + 1];
        auto success = fread(src, 1, size, file);
        src[size] = 0;
        fclose(file);
        if (!success) {
            Log::Error("Failed to load shader file: " + filename);
            delete[] src;
            return "";
        }
        return src;
    }

    void Shader::compile() {
        glCompileShader(id);
    }

    bool Shader::check_compile_errors() {
        int success;
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(id, 512, nullptr, infoLog);
            Log::Error("Shader compilation failed: " + std::string(infoLog));
        }
        return success;
    }

    ComputeShader::ComputeShader() {
        type = GL_COMPUTE_SHADER;
    }

    VertexShader::VertexShader() {
        type = GL_VERTEX_SHADER;
    }

    FragmentShader::FragmentShader() {
        type = GL_FRAGMENT_SHADER;
    }

    GeometryShader::GeometryShader() {
        type = GL_GEOMETRY_SHADER;
    }

    TessControlShader::TessControlShader() {
        type = GL_TESS_CONTROL_SHADER;
    }

    TessEvalShader::TessEvalShader() {
        type = GL_TESS_EVALUATION_SHADER;
    }

    void Program::create() {
        id = glCreateProgram();
    }

    void Program::destroy() {
        glDeleteProgram(id);
    }

    void Program::create_from_source(const std::string &vs, const std::string &fs, const std::string &gs,
                                     const std::string &tc, const std::string &te) {
        if (id == -1) {
            create();
        }
        if (!vs.empty() && !fs.empty()) {
            VertexShader vertexShader;
            vertexShader.create();
            vertexShader.load_source(vs);
            vertexShader.compile();
            vertexShader.check_compile_errors();
            attach(vertexShader);
            vertexShader.destroy();

            FragmentShader fragmentShader;
            fragmentShader.create();
            fragmentShader.load_source(fs);
            fragmentShader.compile();
            fragmentShader.check_compile_errors();
            attach(fragmentShader);
            fragmentShader.destroy();
        }

        if (!gs.empty()) {
            GeometryShader geometryShader;
            geometryShader.create();
            geometryShader.load_source(gs);
            geometryShader.compile();
            geometryShader.check_compile_errors();
            attach(geometryShader);
            geometryShader.destroy();
        }

        if (!tc.empty()) {
            TessControlShader tessControlShader;
            tessControlShader.create();
            tessControlShader.load_source(tc);
            tessControlShader.compile();
            tessControlShader.check_compile_errors();
            attach(tessControlShader);
            tessControlShader.destroy();
        }

        if (!te.empty()) {
            TessEvalShader tessEvalShader;
            tessEvalShader.create();
            tessEvalShader.load_source(te);
            tessEvalShader.compile();
            tessEvalShader.check_compile_errors();
            attach(tessEvalShader);
            tessEvalShader.destroy();
        }

        link();

        if (!check_link_errors()) {
            destroy();
        }
    }

    void Program::create_from_files(const std::string &vs, const std::string &fs, const std::string &gs,
                                    const std::string &tc, const std::string &te) {
        std::string vsSource;
        std::string fsSource;
        std::string gsSource;
        std::string tcSource;
        std::string teSource;

        Shader shader;

        if (!vs.empty() && !fs.empty()) {
            vsSource = shader.load_file(vs);
            fsSource = shader.load_file(fs);
        }

        if (!gs.empty()) {
            gsSource = shader.load_file(gs);
        }

        if (!tc.empty()) {
            tcSource = shader.load_file(tc);
        }

        if (!te.empty()) {
            teSource = shader.load_file(te);
        }

        create_from_source(vsSource, fsSource, gsSource, tcSource, teSource);
    }

    void Program::attach(const Shader &shader) {
        glAttachShader(id, shader.id);
    }

    void Program::detach(const Shader &shader) {
        glDetachShader(id, shader.id);
    }

    void Program::link() {
        glLinkProgram(id);
    }

    bool Program::check_link_errors() {
        int success;
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(id, 512, nullptr, infoLog);
            Log::Error("Program link failed: " + std::string(infoLog));
        }
        return success;
    }

    void Program::use() {
        glUseProgram(id);
    }

    unsigned int Program::get_uniform_location(const std::string &name){
        return glGetUniformLocation(id, name.c_str());
    }

    unsigned int Program::get_uniform_block_index(const std::string &name){
        return glGetUniformBlockIndex(id, name.c_str());
    }

    void Program::bind_uniform_block(const std::string &name, unsigned int binding_point){
        glUniformBlockBinding(id, get_uniform_block_index(name), binding_point);
    }

    bool ComputeShaderProgram::create_from_source(const std::string &cs) {
        if(id == -1) {
            create();
        }
        if (!cs.empty()) {
            ComputeShader computeShader;
            computeShader.create();
            computeShader.load_source(cs);
            computeShader.compile();
            computeShader.check_compile_errors();
            attach(computeShader);
            computeShader.destroy();
        }

        link();

        if (!check_link_errors()) {
            destroy();
            return false;
        }
        return true;
    }

    bool ComputeShaderProgram::create_from_file(const std::string &cs) {
        std::string csSource;
        Shader shader;

        if (!cs.empty()) {
            csSource = shader.load_file(cs);
            if (csSource.empty()) {
                return false;
            }
            create_from_source(csSource);

            if (!check_link_errors()) {
                destroy();
                return false;
            }
            return true;
        }
        return false;
    }

    void
    ComputeShaderProgram::dispatch(unsigned int num_groups_x, unsigned int num_groups_y, unsigned int num_groups_z) {
        glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    }

    void ComputeShaderProgram::memory_barrier(unsigned int barriers) {
        glMemoryBarrier(barriers);
    }
}