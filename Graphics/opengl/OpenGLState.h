//
// Created by alex on 25.07.24.
//

#ifndef ENGINE24_OPENGLSTATE_H
#define ENGINE24_OPENGLSTATE_H

#include "Program.h"
#include "Buffer.h"
#include "VertexArrayObject.h"
#include "entt/fwd.hpp"

namespace Bcg {
    struct OpenGLState {
        OpenGLState(entt::entity entity_id);

        Program get_program(const std::string &name);

        bool register_program(const std::string &name, const Program &program);

        bool remove_program(const std::string &name);

        ComputeShaderProgram get_compute_program(const std::string &name);

        bool register_compute_program(const std::string &name, const ComputeShaderProgram &program);

        bool remove_compute_program(const std::string &name);

        Buffer get_buffer(const std::string &name);

        bool register_buffer(const std::string &name, const Buffer &buffer);

        bool remove_buffer(const std::string &name);

        VertexArrayObject get_vao(const std::string name);

        bool register_vao(const std::string &name, const VertexArrayObject &vao);

        bool remove_vao(const std::string &name);

    private:
        entt::entity entity_id;
    };
}

#endif //ENGINE24_OPENGLSTATE_H
