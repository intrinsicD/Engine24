//
// Created by alex on 25.07.24.
//

#include "OpenGLState.h"
#include "Engine.h"
#include "Logger.h"

namespace Bcg {
    template<typename T>
    struct Manager : public std::unordered_map<std::string, T> {
        using std::unordered_map<std::string, T>::unordered_map;
    };

    struct Programs : public Manager<Program> {
        using Manager<Program>::Manager;
    };

    struct ComputePrograms : public Manager<ComputeShaderProgram> {
        using Manager<ComputeShaderProgram>::Manager;
    };

    struct Buffers : public Manager<Buffer> {
        using Manager<Buffer>::Manager;
    };

    OpenGLState::OpenGLState(entt::entity entity_id) : entity_id(entity_id){
        if(!Engine::Context().find<Programs>()){
            Engine::Context().emplace<Programs>();
        }
        if(!Engine::Context().find<ComputePrograms>()){
            Engine::Context().emplace<ComputePrograms>();
        }
        if(!Engine::State().all_of<Buffers>(entity_id)){
            Engine::State().emplace<Buffers>(entity_id);
        }
    }

    Program OpenGLState::get_program(const std::string &name) {
        auto &programs = Engine::Context().get<Programs>();
        if (programs.find(name) != programs.end()) {
            return programs[name];
        }
        return {};
    }

    bool OpenGLState::register_program(const std::string &name, const Program &program) {
        auto &programs = Engine::Context().get<Programs>();
        if (programs.find(name) != programs.end()) {
            return false;
        }
        programs[name] = program;
        return true;
    }

    bool OpenGLState::remove_program(const std::string &name) {
        auto &programs = Engine::Context().get<Programs>();
        return programs.erase(name);
    }

    ComputeShaderProgram OpenGLState::get_compute_program(const std::string &name){
        auto &programs = Engine::Context().get<ComputePrograms>();
        if (programs.find(name) != programs.end()) {
            return programs[name];
        }
        return {};
    }

    bool OpenGLState::register_compute_program(const std::string &name, const ComputeShaderProgram &program){
        auto &programs = Engine::Context().get<ComputePrograms>();
        if (programs.find(name) != programs.end()) {
            return false;
        }
        programs[name] = program;
        return true;
    }

    bool OpenGLState::remove_compute_program(const std::string &name){
        auto &programs = Engine::Context().get<ComputePrograms>();
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
}