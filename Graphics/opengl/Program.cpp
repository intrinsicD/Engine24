//
// Created by alex on 18.07.24.
//


#include "Program.h"
#include "Logger.h"
#include "glad/gl.h"

namespace Bcg {
    void Program::create() {
        if (id == -1) {
            id = glCreateProgram();
        }
    }

    void Program::destroy() {
        if (id != -1) {
            glDeleteProgram(id);
            id = -1;
        }
    }

    void Program::create_from_source(const std::string &vs_, const std::string &fs_, const std::string &gs_,
                                     const std::string &tc_, const std::string &te_) {
        if (id == -1) {
            create();
        }
        if (!vs_.empty() && !fs_.empty()) {
            vs.create();
            vs.load_source(vs_);
            vs.compile();
            vs.check_compile_errors();
            attach(vs);
            vs.destroy();

            fs.create();
            fs.load_source(fs_);
            fs.compile();
            fs.check_compile_errors();
            attach(fs);
            fs.destroy();
        }

        if (!gs_.empty()) {
            gs.create();
            gs.load_source(gs_);
            gs.compile();
            gs.check_compile_errors();
            attach(gs);
            gs.destroy();
        }

        if (!tc_.empty()) {
            tc.create();
            tc.load_source(tc_);
            tc.compile();
            tc.check_compile_errors();
            attach(tc);
            tc.destroy();
        }

        if (!te_.empty()) {
            te.create();
            te.load_source(te_);
            te.compile();
            te.check_compile_errors();
            attach(te);
            te.destroy();
        }

        link();

        if (!check_link_errors()) {
            destroy();
        }
    }

    void Program::create_from_files(const std::string &vs_, const std::string &fs_, const std::string &gs_,
                                    const std::string &tc_, const std::string &te_) {
        std::string vsSource;
        std::string fsSource;
        std::string gsSource;
        std::string tcSource;
        std::string teSource;

        Shader shader;

        if (!vs_.empty() && !fs_.empty()) {
            vs.filepath = vs_;
            vsSource = shader.load_file(vs_);
            
            fs.filepath = fs_;
            fsSource = shader.load_file(fs_);
        }

        if (!gs_.empty()) {
            gs.filepath = gs_;
            gsSource = shader.load_file(gs_);
        }

        if (!tc_.empty()) {
            tc.filepath = tc_;
            tcSource = shader.load_file(tc_);
        }

        if (!te_.empty()) {
            te.filepath = te_;
            teSource = shader.load_file(te_);
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

    unsigned int Program::get_uniform_location(const std::string &name) {
        return glGetUniformLocation(id, name.c_str());
    }

    unsigned int Program::get_uniform_block_index(const std::string &name) {
        return glGetUniformBlockIndex(id, name.c_str());
    }

    void Program::bind_uniform_block(const std::string &name, unsigned int binding_point) {
        glUniformBlockBinding(id, get_uniform_block_index(name), binding_point);
    }

    void Program::set_uniform1f(const std::string &name, float value){
        int loc = get_uniform_location(name);

        if (loc != -1) {
            glUniform1f(loc, value);
        }
    }

    void Program::set_uniform1ui(const std::string &name, unsigned int value){
        int loc = get_uniform_location(name);

        if (loc != -1) {
            glUniform1ui(loc, value);
        }
    }

    void Program::set_uniform1i(const std::string &name, int value){
        int loc = get_uniform_location(name);

        if (loc != -1) {
            glUniform1i(loc, value);
        }
    }

    void Program::set_uniform3fv(const std::string &name, const float *ptr) {
        int loc = get_uniform_location(name);

        if (loc != -1) {
            glUniform3fv(loc, 1, ptr);
        }
    }

    void Program::set_uniform4fm(const std::string &name, const float *ptr, bool transpose) {
        int loc = get_uniform_location(name);

        if (loc != -1) {
            glUniformMatrix4fv(loc, 1, transpose, ptr);
        }
    }

    bool ComputeShaderProgram::create_from_source(const std::string &cs_) {
        if (id == -1) {
            create();
        }
        if (!cs_.empty()) {
            cs.create();
            cs.load_source(cs_);
            cs.compile();
            cs.check_compile_errors();
            attach(cs);
            cs.destroy();
        }

        link();

        if (!check_link_errors()) {
            destroy();
            return false;
        }
        return true;
    }

    bool ComputeShaderProgram::create_from_file(const std::string &cs_) {
        std::string csSource;
        Shader shader;

        if (!cs_.empty()) {
            cs.filepath = cs_;
            csSource = shader.load_file(cs_);
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