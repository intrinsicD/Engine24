//
// Created by alex on 18.07.24.
//

#include "MeshCommands.h"
#include "Engine.h"
#include "Logger.h"
#include "Mesh.h"
#include "Views.h"
#include "Transform.h"
#include "AABB.h"
#include "Camera.h"
#include "MeshCompute.h"

namespace Bcg::Commands::Mesh {
    void SetupForRendering::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<SurfaceMesh>(entity_id)) {
            Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
            return;
        }

        auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);

        if (!Engine::has<Transform>(entity_id)) {
            Engine::State().emplace<Transform>(entity_id, Transform::Identity());
        }
        if (!Engine::has<AABB>(entity_id)) {
            auto &aabb = Engine::State().emplace<AABB>(entity_id);
            Build(aabb, mesh.positions());
        }

        auto &mw = Engine::State().get_or_emplace<MeshView>(entity_id);

        mw.num_indices = mesh.n_faces() * 3;

        mw.vao.create();
        mw.vao.bind();

        auto v_normals = ComputeVertexNormals(mesh);
        size_t size_in_bytes_vertices = mesh.n_vertices() * sizeof(Point);


        auto &gpu_pos = mw.layout.get_or_add(mesh.vpoint_.name().c_str());
        gpu_pos.size_in_bytes = size_in_bytes_vertices;
        gpu_pos.dims = 3;
        gpu_pos.size = sizeof(float);
        gpu_pos.normalized = false;
        gpu_pos.offset = 0;
        gpu_pos.data = mesh.positions().data();

        auto &gpu_v_normals = mw.layout.get_or_add(v_normals.name().c_str());
        gpu_v_normals.size_in_bytes = size_in_bytes_vertices;
        gpu_v_normals.dims = 3;
        gpu_v_normals.size = sizeof(float);
        gpu_v_normals.normalized = false;
        gpu_v_normals.offset = size_in_bytes_vertices;
        gpu_v_normals.data = v_normals.data();

        mw.vbo.create();
        mw.vbo.bind();
        mw.vbo.buffer_data(nullptr, mw.layout.total_size_bytes(), Buffer::Usage::STATIC_DRAW);
        for (const auto &[name, item]: mw.layout.layout) {
            mw.vbo.buffer_sub_data(item.data, item.size_in_bytes, item.offset);
        }

        auto triangles = extract_triangle_list(mesh);
        mw.ebo.create();
        mw.ebo.bind();
        mw.ebo.buffer_data(triangles.data(), mw.num_indices * sizeof(unsigned int), Buffer::Usage::STATIC_DRAW);

        const char *vertex_shader_src = "#version 330 core\n"
                                        "layout (location = 0) in vec3 aPos;\n"
                                        "layout (location = 1) in vec3 aNormal;\n"
                                        "layout (std140) uniform Camera {\n"
                                        "    mat4 view;\n"
                                        "    mat4 projection;\n"
                                        "};\n"
                                        "out vec3 f_normal;\n"
                                        "out vec3 f_color;\n"
                                        "void main()\n"
                                        "{\n"
                                        "   f_normal = mat3(transpose(inverse(view))) * aNormal;\n"
                                        "   gl_Position = projection * view * vec4(aPos, 1.0);\n"
                                        "}\0";

        const char *fragment_shader_src = "#version 330 core\n"
                                          "in vec3 f_normal;\n"
                                          "out vec4 FragColor;\n"
                                          "uniform vec3 lightDir;\n"
                                          "void main()\n"
                                          "{\n"
                                          "   // Normalize the input normal\n"
                                          "   vec3 normal = normalize(f_normal);\n"
                                          "   // Calculate the diffuse intensity\n"
                                          "   float diff = max(dot(normal, normalize(lightDir)), 0.0);\n"
                                          "   // Define the base color\n"
                                          "   vec3 baseColor = vec3(1.0f, 0.5f, 0.2f);\n"
                                          "   // Calculate the final color\n"
                                          "   vec3 finalColor = diff * baseColor;\n"
                                          "   FragColor = vec4(normal, 1.0f);\n"
                                          "}\n\0";

        mw.program.create_from_source(vertex_shader_src, fragment_shader_src);
        mw.vao.setAttribute(0, 3, VertexArrayObject::AttributeType::FLOAT, false, 3 * sizeof(float), nullptr);
        mw.vao.enableAttribute(0);

        mw.vao.setAttribute(1, 3, VertexArrayObject::AttributeType::FLOAT, false, 3 * sizeof(float),
                            (void *) size_in_bytes_vertices);
        mw.vao.enableAttribute(1);

        // Get the index of the uniform block
        auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
        mw.program.bind_uniform_block("Camera", camera_ubi.binding_point);
        mw.vao.unbind();
        mw.vbo.unbind();


        std::string message = name + ": ";
        message += " #v: " + std::to_string(mesh.n_vertices());
        message += " #e: " + std::to_string(mesh.n_edges());
        message += " #h: " + std::to_string(mesh.n_halfedges());
        message += " #f: " + std::to_string(mesh.n_faces());
        message += " Done.";

        Log::Info(message);
    }
}