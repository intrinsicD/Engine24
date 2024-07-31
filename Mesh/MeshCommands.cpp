//
// Created by alex on 18.07.24.
//

#include "MeshCommands.h"
#include "EntityCommands.h"
#include "Mesh.h"
#include "Views.h"
#include "Transform.h"
#include "AABB.h"
#include "Hierarchy.h"
#include "Camera.h"
#include "CameraCommands.h"
#include "MeshCompute.h"
#include "OpenGLState.h"

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

        if (!Engine::has<AABB>(entity_id)) {
            auto &aabb = Engine::State().emplace<AABB>(entity_id);
            Build(aabb, mesh.positions());
        }


        auto &aabb = Engine::State().get<AABB>(entity_id);

        Vector<float, 3> center = aabb.center();

        for (auto &point: mesh.positions()) {
            point -= center;
        }

        aabb.min -= center;
        aabb.max -= center;

        if (!Engine::has<Transform>(entity_id)) {
            Commands::Entity::Add<Transform>(entity_id, Transform(), "Transform").execute();
        }

        if (!Engine::has<Hierarchy>(entity_id)) {
            Commands::Entity::Add<Hierarchy>(entity_id, Hierarchy(), "Hierarchy").execute();
        }


        auto &mw = Engine::State().get_or_emplace<MeshView>(entity_id);

        mw.num_indices = mesh.n_faces() * 3;

        mw.vao.create();
        mw.vao.bind();

        OpenGLState openGlState(entity_id);

        auto v_normals = ComputeVertexNormals(entity_id, mesh);

        auto b_position = openGlState.get_buffer(mesh.vpoint_.name());
        if (!b_position) {
            b_position = ArrayBuffer();
            b_position.create();
            b_position.bind();
            b_position.buffer_data(mesh.positions().data(),
                                   mesh.positions().size() * 3 * sizeof(float),
                                   Buffer::STATIC_DRAW);
            openGlState.register_buffer(mesh.vpoint_.name(), b_position);
        } else {
            b_position.bind();
        }

        auto &position_attribute = mw.vao.attributes.emplace_back();
        position_attribute.id = 0;
        position_attribute.size = 3;
        position_attribute.type = Attribute::Type::FLOAT;
        position_attribute.normalized = false;
        position_attribute.stride = 3 * sizeof(float);
        position_attribute.shader_name = "positions";
        position_attribute.bound_buffer_name = mesh.vpoint_.name().c_str();
        position_attribute.set(nullptr);
        position_attribute.enable();


        auto b_normals = openGlState.get_buffer(v_normals.name());
        if (!b_normals) {
            b_normals = ArrayBuffer();
            b_normals.create();
            b_normals.bind();
            b_normals.buffer_data(v_normals.data(),
                                  v_normals.vector().size() * 3 * sizeof(float),
                                  Buffer::STATIC_DRAW);
            openGlState.register_buffer(v_normals.name(), b_normals);
        } else {
            b_normals.bind();
        }

        auto &normals_attribute = mw.vao.attributes.emplace_back();
        normals_attribute.id = 1;
        normals_attribute.size = 3;
        normals_attribute.type = Attribute::Type::FLOAT;
        normals_attribute.normalized = false;
        normals_attribute.stride = 3 * sizeof(float);
        normals_attribute.shader_name = "normals";
        normals_attribute.bound_buffer_name = v_normals.name().c_str();
        normals_attribute.set(nullptr);
        normals_attribute.enable();

        auto &colors_attribute = mw.vao.attributes.emplace_back();
        colors_attribute.id = 2;
        colors_attribute.size = 3;
        colors_attribute.type = Attribute::Type::FLOAT;
        colors_attribute.normalized = false;
        colors_attribute.stride = 3 * sizeof(float);
        colors_attribute.shader_name = "colors";
        colors_attribute.bound_buffer_name = v_normals.name().c_str();
        colors_attribute.set(nullptr);
        colors_attribute.enable();
        colors_attribute.set_default(mw.base_color.data());

        auto f_triangles = extract_triangle_list(mesh);
        auto b_triangles = openGlState.get_buffer(f_triangles.name());
        if (!b_triangles) {
            b_triangles = ElementArrayBuffer();
            b_triangles.create();
            b_triangles.bind();
            b_triangles.buffer_data(f_triangles.data(),
                                    f_triangles.vector().size() * 3 * sizeof(unsigned int),
                                    Buffer::STATIC_DRAW);
            openGlState.register_buffer(f_triangles.name(), b_triangles);
        } else {
            b_triangles.bind();
        }

        const char *vertex_shader_src = "#version 330 core\n"
                                        "layout (location = 0) in vec3 aPos;\n"
                                        "layout (location = 1) in vec3 aNormal;\n"
                                        "layout (location = 2) in vec3 aColor;\n"
                                        "layout (std140) uniform Camera {\n"
                                        "    mat4 view;\n"
                                        "    mat4 projection;\n"
                                        "};\n"
                                        "uniform mat4 model;\n"
                                        "out vec3 f_normal;\n"
                                        "out vec3 f_world;\n"
                                        "out vec3 f_color;\n"
                                        "void main()\n"
                                        "{\n"
                                        "   f_normal = mat3(transpose(inverse(model))) * aNormal;\n"
                                        "   f_color = aColor;"
                                        "   f_world = (model * vec4(aPos, 1.0)).xyz;"
                                        "   gl_Position = projection * view * vec4(f_world, 1.0);\n"
                                        "}\0";

        const char *fragment_shader_src = "#version 330 core\n"
                                          "in vec3 f_normal;\n"
                                          "in vec3 f_color;\n"
                                          "in vec3 f_world;\n"
                                          "out vec4 FragColor;\n"
                                          "uniform vec3 lightPosition;\n"
                                          "void main()\n"
                                          "{\n"
                                          "   // Normalize the input normal\n"
                                          "   vec3 normal = normalize(f_normal);\n"
                                          "   // Calculate the diffuse intensity\n"
                                          "   float diff = max(dot(normal, normalize(lightPosition - f_world)), 0.01);\n"
                                          "   // Calculate the final color\n"
                                          "   vec3 finalColor = diff * f_color;\n"
                                          "   FragColor = vec4(finalColor, 1.0f);\n"
                                          "}\n\0";

        auto program = openGlState.get_program("MeshProgram");
        if (!program) {
            program.create_from_source(vertex_shader_src, fragment_shader_src);
            openGlState.register_program("MeshProgram", program);
        }
        mw.program = program;


        // Get the index of the uniform block
        auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
        mw.program.bind_uniform_block("Camera", camera_ubi.binding_point);
        mw.vao.unbind();

        b_position.unbind();
        b_normals.unbind();
        b_triangles.unbind();

        std::string message = name + ": ";
        message += " #v: " + std::to_string(mesh.n_vertices());
        message += " #e: " + std::to_string(mesh.n_edges());
        message += " #h: " + std::to_string(mesh.n_halfedges());
        message += " #f: " + std::to_string(mesh.n_faces());
        message += " Done.";

        Log::Info(message);
        CenterCamera(entity_id).execute();
    }

    void ComputeFaceNormals::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<SurfaceMesh>(entity_id)) {
            Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
            return;
        }

        auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);

        auto v_normals = ComputeVertexNormals(entity_id, mesh);
    }
}