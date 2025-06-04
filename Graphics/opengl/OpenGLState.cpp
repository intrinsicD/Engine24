//
// Created by alex on 25.07.24.
//

#include "OpenGLState.h"
#include "Engine.h"
#include "FileWatcher.h"
#include "Logger.h"

namespace Bcg {
    template<typename T>
    struct Cache : public std::unordered_map<std::string, T> {
        using std::unordered_map<std::string, T>::unordered_map;
    };

    struct Programs : public Cache<Program> {
        using Cache<Program>::Cache;
    };

    struct ComputePrograms : public Cache<ComputeShaderProgram> {
        using Cache<ComputeShaderProgram>::Cache;
    };

    struct Buffers : public Cache<Buffer> {
        using Cache<Buffer>::Cache;
    };

    struct VertexArrayObjects : public Cache<VertexArrayObject> {
        using Cache<VertexArrayObject>::Cache;
    };

    OpenGLState::OpenGLState(entt::entity entity_id) : entity_id(entity_id) {
        if (!Engine::Context().find<Programs>()) {
            Engine::Context().emplace<Programs>();
        }
        if (!Engine::Context().find<ComputePrograms>()) {
            Engine::Context().emplace<ComputePrograms>();
        }
        if (!Engine::Context().find<FileWatcher>()) {
            Engine::Context().emplace<FileWatcher>();
        }
        if (!Engine::State().all_of<Buffers>(entity_id)) {
            Engine::State().emplace<Buffers>(entity_id);
        }

        if (!Engine::State().all_of<VertexArrayObjects>(entity_id)) {
            Engine::State().emplace<VertexArrayObjects>(entity_id);
        }
    }

    Program OpenGLState::get_program(const std::string &name) {
        auto &programs = Engine::Context().get<Programs>();
        if (programs.find(name) != programs.end()) {
            return programs[name];
        }
        return {};
    }

    static void reload_program(const std::string &name, Shader &shader, Program &program) {
        auto source = shader.load_file(shader.filepath);
        auto old_source = shader.source;
        shader.load_source(source);
        shader.compile();
        if (shader.check_compile_errors()) {
            program.attach(shader);
            shader.destroy();
            program.link();
            if (program.check_link_errors()) {
                return;
            } else {
                Log::Error("Failed to link program: " + name);
            }
        } else {
            Log::Error("Failed to compile shader: " + shader.filepath);
        }
        Log::Info("Recompile shader: " + shader.filepath);
        shader.load_source(old_source);
        shader.compile();
        shader.check_compile_errors();
        program.attach(shader);
        shader.destroy();
        program.link();
    }

    bool OpenGLState::register_program(const std::string &name, const Program &program) {
        auto &programs = Engine::Context().get<Programs>();
        if (programs.find(name) != programs.end()) {
            return false;
        }
        programs[name] = program;

        auto &watcher = Engine::Context().get<FileWatcher>();

        File vs_file = {program.vs.filepath.c_str(), [name]() {
            auto &programs = Engine::Context().get<Programs>();
            auto &program = programs[name];
            reload_program(name, program.vs, program);
        }};

        File fs_file = {program.fs.filepath.c_str(), [name]() {
            auto &programs = Engine::Context().get<Programs>();
            auto &program = programs[name];
            reload_program(name, program.fs, program);
        }};

        watcher.watch(vs_file);
        watcher.watch(fs_file);

        if (program.gs) {
            File gs_file = {program.gs.filepath.c_str(), [name]() {
                auto &programs = Engine::Context().get<Programs>();
                auto &program = programs[name];
                reload_program(name, program.gs, program);
            }};
            watcher.watch(gs_file);
        }

        if (program.tc) {
            File tc_file = {program.tc.filepath.c_str(), [name]() {
                auto &programs = Engine::Context().get<Programs>();
                auto &program = programs[name];
                reload_program(name, program.tc, program);
            }};
            watcher.watch(tc_file);
        }

        if (program.te) {
            File te_file = {program.te.filepath.c_str(), [name]() {
                auto &programs = Engine::Context().get<Programs>();
                auto &program = programs[name];
                reload_program(name, program.te, program);
            }};
            watcher.watch(te_file);
        }
        return true;
    }

    bool OpenGLState::remove_program(const std::string &name) {
        auto &programs = Engine::Context().get<Programs>();
        auto &watcher = Engine::Context().get<FileWatcher>();
        auto &program = programs[name];

        watcher.remove(program.vs.filepath);
        watcher.remove(program.fs.filepath);

        if (program.gs) {
            watcher.remove(program.gs.filepath);
        }

        if (program.tc) {
            watcher.remove(program.tc.filepath);
        }

        if (program.te) {
            watcher.remove(program.te.filepath);
        }
        return programs.erase(name);
    }

    ComputeShaderProgram OpenGLState::get_compute_program(const std::string &name) {
        auto &programs = Engine::Context().get<ComputePrograms>();
        if (programs.find(name) != programs.end()) {
            return programs[name];
        }
        return {};
    }

    bool OpenGLState::register_compute_program(const std::string &name, const ComputeShaderProgram &program) {
        auto &programs = Engine::Context().get<ComputePrograms>();
        if (programs.find(name) != programs.end()) {
            return false;
        }
        programs[name] = program;

        auto &watcher = Engine::Context().get<FileWatcher>();

        if (program.cs) {
            File cs_file = {program.cs.filepath, [name]() {
                auto &programs = Engine::Context().get<Programs>();
                auto &program = programs[name];
                reload_program(name, program.cs, program);
            }};
            watcher.watch(cs_file);
        }
        return true;
    }

    bool OpenGLState::remove_compute_program(const std::string &name) {
        auto &programs = Engine::Context().get<ComputePrograms>();
        auto &watcher = Engine::Context().get<FileWatcher>();
        auto &program = programs[name];

        watcher.remove(program.cs.filepath);
        return programs.erase(name);
    }

    Buffer OpenGLState::get_buffer(const std::string &name) {
        auto &buffers = Engine::State().get<Buffers>(entity_id);
        if (buffers.find(name) != buffers.end()) {
            return buffers[name];
        }
        return {};
    }

    bool OpenGLState::register_buffer(const std::string &name, const Buffer &buffer) {
        auto &buffers = Engine::State().get<Buffers>(entity_id);
        if (buffers.find(name) != buffers.end()) {
            return false;
        }
        buffers[name] = buffer;
        return true;
    }

    bool OpenGLState::remove_buffer(const std::string &name) {
        auto &buffers = Engine::State().get<Buffers>(entity_id);
        return buffers.erase(name);
    }

    VertexArrayObject OpenGLState::get_vao(const std::string name) {
        auto &vaos = Engine::State().get<VertexArrayObjects>(entity_id);
        if (vaos.find(name) != vaos.end()) {
            return vaos[name];
        }
        return {};
    }

    bool OpenGLState::register_vao(const std::string &name, const Bcg::VertexArrayObject &vao) {
        auto &vaos = Engine::State().get<VertexArrayObjects>(entity_id);
        if (vaos.find(name) != vaos.end()) {
            return false;
        }
        vaos[name] = vao;
        return true;
    }

    bool OpenGLState::remove_vao(const std::string &name) {
        auto &vaos = Engine::State().get<VertexArrayObjects>(entity_id);
        return vaos.erase(name);
    }

    void OpenGLState::clear() {
        auto &vaos = Engine::State().get<VertexArrayObjects>(entity_id);
        for (auto &vao: vaos) {
            vao.second.destroy();
        }
        auto &buffers = Engine::State().get<Buffers>(entity_id);
        for (auto &buffer: buffers) {
            buffer.second.destroy();
        }
    }
}