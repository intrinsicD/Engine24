//
// Created by alex on 30.07.24.
//

#include "PointCloudCommands.h"
#include "EntityCommands.h"
#include "PointCloud.h"
#include "Views.h"
#include "Transform.h"
#include "AABB.h"
#include "Hierarchy.h"
#include "Camera.h"
#include "CameraCommands.h"
#include "PointCloudCompute.h"
#include "KDTreeCompute.h"
#include "OpenGLState.h"
#include <numeric>

namespace Bcg::Commands::Points {
    void SetupForRendering::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<PointCloud>(entity_id)) {
            Log::Warn(name + "Entity does not have a PointCloud. Abort Command");
            return;
        }

        auto &pc = Engine::State().get<PointCloud>(entity_id);

        if (!Engine::has<AABB>(entity_id)) {
            auto &aabb = Engine::State().emplace<AABB>(entity_id);
            Build(aabb, pc.positions());
        }


        auto &aabb = Engine::State().get<AABB>(entity_id);

        Vector<float, 3> center = aabb.center();

        for (auto &point: pc.positions()) {
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


        auto &pcw = Engine::State().get_or_emplace<PointCloudView>(entity_id);

        pcw.offset = 0;
        pcw.num_indices = pc.n_vertices();

        pcw.vao.create();
        pcw.vao.bind();

        OpenGLState openGlState(entity_id);

        BuildKDTReeCompute(entity_id, pc.vpoint_);
        auto v_normals = ComputeVertexNormals(entity_id, pc.vprops_, 10);

        auto b_position = openGlState.get_buffer(pc.vpoint_.name());
        if (!b_position) {
            b_position = ArrayBuffer();
            b_position.create();
            b_position.bind();
            b_position.buffer_data(pc.positions().data(),
                                   pc.positions().size() * 3 * sizeof(float),
                                   Buffer::STATIC_DRAW);
            openGlState.register_buffer(pc.vpoint_.name(), b_position);
        } else {
            b_position.bind();
        }

        pcw.vao.setAttribute(0, 3, VertexArrayObject::AttributeType::FLOAT, false, 3 * sizeof(float), nullptr);
        pcw.vao.enableAttribute(0);

        auto v_indices = pc.vertex_property<unsigned int>("v:indices", 0);
        std::iota(v_indices.vector().begin(), v_indices.vector().end(), 0);

        auto b_indices = openGlState.get_buffer(v_indices.name());
        if (!b_indices) {
            b_indices = ElementArrayBuffer();
            b_indices.create();
            b_indices.bind();
            b_indices.buffer_data(v_indices.data(),
                                  v_indices.vector().size() * sizeof(unsigned int),
                                  Buffer::STATIC_DRAW);
            openGlState.register_buffer(v_indices.name(), b_indices);
        } else {
            b_indices.bind();
        }

        const char *vertex_shader_src = "#version 330 core\n"
                                        "layout (location = 0) in vec3 aPos;\n"
                                        "layout (location = 1) in vec3 aNormal;\n"
                                        "layout (std140) uniform Camera {\n"
                                        "    mat4 view;\n"
                                        "    mat4 projection;\n"
                                        "};\n"
                                        "uniform mat4 model;\n"
                                        "out vec3 f_normal;\n"
                                        "out vec3 f_color;\n"
                                        "void main()\n"
                                        "{\n"
                                        "   mat4 vm = view * model;"
                                        "   f_normal = mat3(transpose(inverse(vm))) * aNormal;\n"
                                        "   gl_Position = projection * vm * vec4(aPos, 1.0);\n"
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

        auto program = openGlState.get_program("PointCloudProgram");
        if (!program) {
            program.create_from_source(vertex_shader_src, fragment_shader_src);
            openGlState.register_program("PointCloudProgram", program);
        }
        pcw.program = program;


        // Get the index of the uniform block
        auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
        pcw.program.bind_uniform_block("Camera", camera_ubi.binding_point);
        pcw.vao.unbind();

        b_position.unbind();
        b_indices.unbind();

        std::string message = name + ": ";
        message += " #v: " + std::to_string(pc.n_vertices());
        message += " Done.";

        Log::Info(message);
        CenterCamera(entity_id).execute();
    }
}