//
// Created by alex on 04.08.24.
//


#include "MeshViewCommands.h"
#include "Engine.h"
#include "MeshView.h"
#include "Mesh.h"
#include "GetPrimitives.h"
#include "OpenGLState.h"
#include "Camera.h"
#include "Logger.h"
#include "PropertyEigenMap.h"

namespace Bcg::Commands::View {
    void SetupMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        auto *faces = GetPrimitives(entity_id).faces();
        if (!vertices) return;

        auto &view = Engine::require<MeshView>(entity_id);
        size_t num_faces = faces->size();
        view.num_indices = num_faces * 3;

        OpenGLState openGlState(entity_id);

        auto program = openGlState.get_program("MeshProgram");
        if (!program) {
            program.create_from_files("../Shaders/glsl/surface_mesh_vs.glsl",
                                      "../Shaders/glsl/surface_mesh_fs.glsl");

            auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
            view.program.bind_uniform_block("Camera", camera_ubi.binding_point);

            openGlState.register_program("MeshProgram", program);
        }

        view.program = program;

        view.vao.create();

        SetPositionMeshView(entity_id, "v:point").execute();
        SetNormalMeshView(entity_id, "v:normal").execute();
        SetColorMeshView(entity_id, "uniform_color").execute();

        auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
        auto f_triangles = extract_triangle_list(mesh);
        SetTrianglesMeshView(entity_id, f_triangles.vector()).execute();

        view.vao.unbind();
    }

    void SetPositionMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            SetupMeshView(entity_id).execute();
        }

        auto &view = Engine::require<MeshView>(entity_id);
        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto v_positions = vertices->get<Vector<float, 3>>(property_name);
        auto b_position = openGlState.get_buffer(property_name);

        if (v_positions) {
            if (!b_position) {
                b_position = ArrayBuffer();
                b_position.create();
                openGlState.register_buffer(property_name, b_position);
            }

            view.vao.bind();
            b_position.bind();
            b_position.buffer_data(v_positions.data(),
                                   num_vertices * 3 * sizeof(float),
                                   Buffer::STATIC_DRAW);

            view.position.bound_buffer_name = property_name.c_str();
            view.position.set(nullptr);
            view.position.enable();
            view.vao.unbind();
            b_position.unbind();
        } else {
            Log::Warn(name + ": failed, because entity does not have " + property_name + " property.");
        }
    }

    void SetNormalMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            SetupMeshView(entity_id).execute();
        }

        auto &view = Engine::require<MeshView>(entity_id);
        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto v_normals = vertices->get<Vector<float, 3>>(property_name);
        auto b_normals = openGlState.get_buffer(property_name);

        if (v_normals) {
            if (!b_normals) {
                b_normals = ArrayBuffer();
                b_normals.create();
                openGlState.register_buffer(property_name, b_normals);
            }

            view.vao.bind();
            b_normals.bind();
            b_normals.buffer_data(v_normals.data(),
                                  num_vertices * 3 * sizeof(float),
                                  Buffer::STATIC_DRAW);

            view.normal.bound_buffer_name = property_name.c_str();
            view.normal.set(nullptr);
            view.normal.enable();
            view.vao.unbind();
            b_normals.unbind();
        } else {
            Log::Warn(name + ": failed, because entity does not have " + property_name + " property.");
        }
    }

    void SetColorMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            SetupMeshView(entity_id).execute();
        }

        auto &view = Engine::require<MeshView>(entity_id);
        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto v_color = vertices->get<Vector<float, 3>>(property_name);
        auto b_color = openGlState.get_buffer(property_name);

        if (v_color) {
            if (!b_color) {
                b_color = ArrayBuffer();
                b_color.create();
                openGlState.register_buffer(property_name, b_color);
            }

            view.vao.bind();
            b_color.bind();
            b_color.buffer_data(v_color.data(),
                                num_vertices * 3 * sizeof(float),
                                Buffer::STATIC_DRAW);
            view.min_color = Map(v_color.vector()).minCoeff();
            view.max_color = Map(v_color.vector()).maxCoeff();

            view.color.bound_buffer_name = property_name.c_str();
            view.color.set(nullptr);
            view.color.enable();
            view.use_uniform_color = false;
        } else {
            view.vao.bind();
            view.color.bound_buffer_name = "uniform_color";
            view.color.disable();
            view.color.set_default(view.uniform_color.data());
            view.use_uniform_color = true;
        }
        view.vao.unbind();

        if (v_color) {
            b_color.unbind();
        }
    }

    void SetTrianglesMeshView::execute() const {
        if (tris.empty()) {
            Log::Error(name + ": failed, tris vector is empty!");
            return;
        }

        auto *faces = GetPrimitives(entity_id).faces();
        if (!faces) return;

        if (!Engine::has<MeshView>(entity_id)) {
            SetupMeshView(entity_id).execute();
        }

        auto &view = Engine::require<MeshView>(entity_id);
        size_t num_faces = faces->size();

        OpenGLState openGlState(entity_id);

        auto b_indices = openGlState.get_buffer("f:indices");
        if (!b_indices) {
            b_indices = ElementArrayBuffer();
            b_indices.create();
            openGlState.register_buffer("f:indices", b_indices);
        }

        b_indices = openGlState.get_buffer("f:indices");

        view.vao.bind();
        b_indices.bind();
        b_indices.buffer_data(tris.data(),
                              num_faces * 3 * sizeof(unsigned int),
                              Buffer::STATIC_DRAW);
        view.vao.unbind();
        b_indices.unbind();
    }
}