//
// Created by alex on 21.06.24.
//

#include "MeshCompute.h"
#include "glad/gl.h"
#include "Logger.h"
#include <string>

namespace Bcg {

    unsigned int CompileComputeShader(const char *source) {
        unsigned int program = glCreateProgram();
        unsigned int shader = glCreateShader(GL_COMPUTE_SHADER);

        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            Log::Error(("Compute shader compilation failed\n" + std::string(infoLog)).c_str());
        }

        glAttachShader(program, shader);
        glLinkProgram(program);

        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            Log::Error(("Compute shader program linking failed\n" + std::string(infoLog)).c_str());
        }

        glDeleteShader(shader);
        return program;
    }

    const char *computeShaderSource = R"(
        #version 430 core
        layout (local_size_x = 1) in;
        layout (std430, binding = 0) readonly buffer VertexPositions { vec4 positions[]; };
        layout (std430, binding = 1) readonly buffer Indices { uvec3 indices[]; };
        layout (std430, binding = 2) writeonly buffer FaceNormals { vec4 normals[]; };
        void main() {
            uint index = gl_GlobalInvocationID.x;
            uvec3 vertexIndices = indices[index];
            vec3 v0 = positions[vertexIndices.x].xyz;
            vec3 v1 = positions[vertexIndices.y].xyz;
            vec3 v2 = positions[vertexIndices.z].xyz;
            normals[index].xyz = normalize(cross(v1 - v0, v2 - v0));
        }
    )";

    std::vector<float> ComputeFaceNormals(MeshComponent &mesh) {
        GLuint vertexBuffer, indexBuffer, normalBuffer;

// Vertex buffer (single precision positions)
        glGenBuffers(1, &vertexBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, mesh.positions.size() * sizeof(mesh.positions[0]), mesh.positions.data(),
                     GL_STATIC_DRAW);

// Index buffer (unsigned int indices)
        glGenBuffers(1, &indexBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, mesh.triangles.size() * sizeof(mesh.triangles[0]), mesh.triangles.data(),
                     GL_STATIC_DRAW);

// Normal buffer (single precision normals)
        std::vector<float> result(mesh.triangles.size() / 3 * 4);
        glGenBuffers(1, &normalBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, normalBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, result.size() * sizeof(result[0]), nullptr,
                     GL_DYNAMIC_DRAW);

        auto program = CompileComputeShader(computeShaderSource);
        if (program == 0) {
            Log::Error("Failed to create compute shader program!\n");
            return {};
        }
        glUseProgram(program);

// Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, normalBuffer);

// Dispatch compute shader
        glDispatchCompute(mesh.triangles.size() / 3, 1, 1);

// Ensure all computations are done
        //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

// Retrieve the computed normals
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, normalBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result.size() * sizeof(result[0]), result.data());
// Process the normals...
        glDeleteBuffers(1, &vertexBuffer);
        glDeleteBuffers(1, &indexBuffer);
        glDeleteBuffers(1, &normalBuffer);
        glDeleteProgram(program);
        return result;
    }
}