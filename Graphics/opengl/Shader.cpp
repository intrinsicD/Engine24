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

    std::string Shader::load_file(const std::string &filepath) {
        FILE *file = fopen(filepath.c_str(), "r");
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
            Log::Error("Failed to load shader file: " + filepath);
            delete[] src;
            return "";
        }
        this->filepath = filepath;
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
}