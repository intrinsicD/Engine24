//
// Created by alex on 21.06.24.
//

#include "MeshCompute.h"
#include "Mesh.h"
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

    pmp::VertexProperty<pmp::Vector<float, 3>> ComputeVertexNormals(pmp::SurfaceMesh &mesh) {
        const char *computeShaderSource = R"(
        #version 430 core
        layout (local_size_x = 1) in;
        layout (std430, binding = 0) readonly buffer VertexPositions { vec4 positions[]; };
        layout (std430, binding = 1) readonly buffer VertexConnectivity { uint vconn[]; };
        layout (std430, binding = 2) readonly buffer HalfedgeConnectivity { uvec4 hconn[]; };
        layout (std430, binding = 3) readonly buffer FaceConnectivity { uint fconn[]; };
        layout (std430, binding = 4) writeonly buffer FaceNormals { vec4 normals[]; };

        uint face(uint h) {
            return hconn[h].x;
        }

        uint to_vertex(uint h) {
            return hconn[h].y;
        }

        uint next_halfedge(uint h) {
            return hconn[h].z;
        }

        uint prev_halfedge(uint h) {
            return hconn[h].w;
        }

        uint opposite_halfedge(uint h) {
            return ((h & 1) == 1 ? h - 1 : h + 1);
        }

        uint ccw_rotated_halfedge(uint h) {
           return opposite_halfedge(prev_halfedge(h));
        }

        uint cw_rotated_halfedge(uint h) {
            return next_halfedge(opposite_halfedge(h));
        }

        void main() {
            uint v = gl_GlobalInvocationID.x;
            uint h = vconn[v];
            uint start = h;

            vec3 normal = vec3(0.0);
            vec3 v0 = positions[v].xyz;

             do {
                    uint nh = next_halfedge(h);
                    vec3 v1 = positions[to_vertex(h)].xyz;
                    vec3 v2 = positions[to_vertex(nh)].xyz;
                    normal += normalize(cross(v1 - v0, v2 - v0));
                    h = cw_rotated_halfedge(h);
                } while (h != start && h != uint(-1));

            float length = length(v0);
            normals[v] = vec4(normalize(normal), v);
        }
    )";
        // point coordinates
        auto &vpoint = mesh.vpoint_.vector();
        auto &vconn = mesh.vconn_.vector();
        auto &hconn = mesh.hconn_.vector();
        auto &fconn = mesh.fconn_.vector();
        auto normals = mesh.add_vertex_property<pmp::Vector<float, 3>>("v:normal");

        Eigen::Matrix<float, 4, -1> P = Eigen::Matrix<float, 4, -1>::Zero(4, vpoint.size());
        P.block(0, 0, 3, vpoint.size()) = Eigen::Map<Eigen::Matrix<float, 3, -1>>(vpoint[0].data(), 3, vpoint.size());

        GLuint vpointBuffer, vconnBuffer, hconnBuffer, fconnBuffer, normalBuffer;
        glGenBuffers(1, &vpointBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vpointBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, P.size() * sizeof(float), P.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &vconnBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vconnBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, vconn.size() * sizeof(unsigned int), vconn.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &hconnBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, hconnBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, hconn.size() * 4 * sizeof(unsigned int), hconn.data(),
                     GL_STATIC_DRAW);

        glGenBuffers(1, &fconnBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, fconnBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, fconn.size() * sizeof(unsigned int), fconn.data(),
                     GL_STATIC_DRAW);

        std::vector<pmp::Vector<float, 4>> result(mesh.vertices_size());
        glGenBuffers(1, &normalBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, normalBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, result.size() * 4 * sizeof(float), nullptr,
                     GL_DYNAMIC_DRAW);

        auto program = CompileComputeShader(computeShaderSource);
        if (program == 0) {
            Log::Error("Failed to create compute shader program!\n");
            return normals;
        }
        glUseProgram(program);

// Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vpointBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vconnBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, hconnBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, fconnBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, normalBuffer);

// Dispatch compute shader
        glDispatchCompute(mesh.vertices_size(), 1, 1);

// Ensure all computations are done
        //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

// Retrieve the computed normals
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, normalBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result.size() * 4 * sizeof(float), result.data());
// Process the normals...
        glDeleteBuffers(1, &vpointBuffer);
        glDeleteBuffers(1, &vconnBuffer);
        glDeleteBuffers(1, &hconnBuffer);
        glDeleteBuffers(1, &fconnBuffer);
        glDeleteBuffers(1, &normalBuffer);
        glDeleteProgram(program);

        for (size_t i = 0; i < result.size(); ++i) {
            normals[pmp::Vertex(i)][0] = result[i][0];
            normals[pmp::Vertex(i)][1] = result[i][1];
            normals[pmp::Vertex(i)][2] = result[i][2];
           // std::cout << result[i][0] << ", " << result[i][1] << ", " << result[i][2] << "  v:" << result[i][3] << "\n";
        }
        return normals;
    }


    pmp::FaceProperty<pmp::Vector<float, 3>> ComputeFaceNormals(pmp::SurfaceMesh &mesh) {
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

        std::vector<float> positions;
        std::vector<unsigned int> triangles;
        auto normals = mesh.add_face_property<pmp::Vector<float, 3>>("f:normal");
        extract_triangle_list(mesh, positions, triangles);

        GLuint vertexBuffer, indexBuffer, normalBuffer;

// Vertex buffer (single precision positions)
        glGenBuffers(1, &vertexBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, vertexBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, positions.size() * sizeof(positions[0]), positions.data(),
                     GL_STATIC_DRAW);

// Index buffer (unsigned int indices)
        glGenBuffers(1, &indexBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, triangles.size() * sizeof(triangles[0]), triangles.data(),
                     GL_STATIC_DRAW);

// Normal buffer (single precision normals)
        std::vector<float> result(triangles.size() / 3 * 4);
        glGenBuffers(1, &normalBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, normalBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, result.size() * sizeof(result[0]), nullptr,
                     GL_DYNAMIC_DRAW);

        auto program = CompileComputeShader(computeShaderSource);
        if (program == 0) {
            Log::Error("Failed to create compute shader program!\n");
            return normals;
        }
        glUseProgram(program);

// Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, normalBuffer);

// Dispatch compute shader
        glDispatchCompute(triangles.size() / 3, 1, 1);

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

        for (size_t i = 0; i < result.size() / 4; ++i) {
            pmp::vec3 normal(result[i * 4], result[i * 4 + 1], result[i * 4 + 2]);
            normals[pmp::Face(i)] = normal;
        }
        return normals;
    }
}