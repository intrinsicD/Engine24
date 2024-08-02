//
// Created by alex on 01.08.24.
//

#include "ViewCommands.h"
#include "Views.h"
#include "Engine.h"
#include "Camera.h"
#include "Mesh.h"
#include "MeshCompute.h"
#include "OpenGLState.h"
#include "GLUtils.h"
#include "GetPrimitives.h"
#include "FileWatcher.h"
#include <numeric>

namespace Bcg::Commands::View {
    void SetupPointsView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        auto &pcw = Engine::require<PointCloudView>(entity_id);
        size_t num_vertices = vertices->size();
        pcw.offset = 0;
        pcw.num_indices = num_vertices;

        pcw.vao.create();
        pcw.vao.bind();

        OpenGLState openGlState(entity_id);

        auto positions = vertices->get<PointType>("v:point");
        auto b_position = openGlState.get_buffer(positions.name());
        if (!b_position) {
            b_position = ArrayBuffer();
            b_position.create();
            b_position.bind();
            b_position.buffer_data(positions.data(),
                                   num_vertices * 3 * sizeof(float),
                                   Buffer::STATIC_DRAW);
            openGlState.register_buffer(positions.name(), b_position);
        } else {
            b_position.bind();
        }

        auto &position_attribute = pcw.vao.attributes.emplace_back();
        position_attribute.id = BCG_GL_DEFAULT_POSITION_LOC;
        position_attribute.size = 3;
        position_attribute.type = Attribute::Type::FLOAT;
        position_attribute.normalized = false;
        position_attribute.stride = 3 * sizeof(float);
        position_attribute.shader_name = "positions";
        position_attribute.bound_buffer_name = positions.name().c_str();
        position_attribute.set(nullptr);
        position_attribute.enable();

        auto v_indices = vertices->get_or_add<unsigned int>("v:indices", 0);
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

        pcw.vao.unbind();

        b_position.unbind();
        b_indices.unbind();

        pcw.vao.bind();
        auto &normal_attribute = pcw.vao.attributes.emplace_back();
        normal_attribute.id = BCG_GL_DEFAULT_NORMAL_LOC;
        normal_attribute.size = 3;
        normal_attribute.type = Attribute::Type::FLOAT;
        normal_attribute.normalized = false;
        normal_attribute.stride = 3 * sizeof(float);
        normal_attribute.shader_name = "normals";
        normal_attribute.bound_buffer_name = "";
        normal_attribute.set(nullptr);

        auto &colors_attribute = pcw.vao.attributes.emplace_back();
        colors_attribute.id = BCG_GL_DEFAULT_COLOR_LOC;
        colors_attribute.size = 3;
        colors_attribute.type = Attribute::Type::FLOAT;
        colors_attribute.normalized = false;
        colors_attribute.stride = 3 * sizeof(float);
        colors_attribute.shader_name = "colors";
        colors_attribute.bound_buffer_name = "";
        colors_attribute.set(nullptr);
        colors_attribute.set_default(pcw.base_color.data());
        pcw.vao.unbind();

        auto program = openGlState.get_program("PointCloudProgram");
        if (!program) {
            program.create_from_files("../Shaders/glsl/impostor_spheres_vs.glsl", "../Shaders/glsl/impostor_spheres_fs.glsl");

            auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
            pcw.program.bind_uniform_block("Camera", camera_ubi.binding_point);

            openGlState.register_program("PointCloudProgram", program);
        }
        pcw.program = program;
    }

    void SetupGraphView::execute() const {

    }

    void SetupMeshView::execute() const {
        if (!Engine::valid(entity_id)) return;
        if (!Engine::has<SurfaceMesh>(entity_id)) return;

        auto &mw = Engine::require<MeshView>(entity_id);
        auto &mesh = Engine::require<SurfaceMesh>(entity_id);

        mw.num_indices = mesh.n_faces() * 3;

        mw.vao.create();
        mw.vao.bind();

        OpenGLState openGlState(entity_id);

        auto v_normals = ComputeSurfaceMeshVertexNormals(entity_id);

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
        position_attribute.id = BCG_GL_DEFAULT_POSITION_LOC;
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
        normals_attribute.id = BCG_GL_DEFAULT_NORMAL_LOC;
        normals_attribute.size = 3;
        normals_attribute.type = Attribute::Type::FLOAT;
        normals_attribute.normalized = false;
        normals_attribute.stride = 3 * sizeof(float);
        normals_attribute.shader_name = "normals";
        normals_attribute.bound_buffer_name = v_normals.name().c_str();
        normals_attribute.set(nullptr);
        normals_attribute.enable();

        auto &colors_attribute = mw.vao.attributes.emplace_back();
        colors_attribute.id = BCG_GL_DEFAULT_COLOR_LOC;
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

        mw.vao.unbind();

        b_position.unbind();
        b_normals.unbind();
        b_triangles.unbind();

        auto program = openGlState.get_program("MeshProgram");
        if (!program) {
            program.create_from_files("../Shaders/glsl/surface_mesh_vs.glsl", "../Shaders/glsl/surface_mesh_fs.glsl");
            // Get the index of the uniform block
            auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
            program.bind_uniform_block("Camera", camera_ubi.binding_point);

            openGlState.register_program("MeshProgram", program);
        }
        mw.program = program;
    }
}