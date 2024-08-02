//
// Created by alex on 02.08.24.
//

#include "SphereViewCommands.h"
#include "Engine.h"
#include "SphereView.h"
#include "GetPrimitives.h"
#include "OpenGLState.h"
#include "Camera.h"
#include "Logger.h"
#include "PropertyEigenMap.h"
#include <numeric>

namespace Bcg::Commands::View {
    void SetupSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();
        view.num_spheres = num_vertices;

        OpenGLState openGlState(entity_id);

        auto program = openGlState.get_program("SpheresProgram");
        if (!program) {
            program.create_from_files("../Shaders/glsl/impostor_spheres_vs.glsl",
                                      "../Shaders/glsl/impostor_spheres_fs.glsl");

            auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
            view.program.bind_uniform_block("Camera", camera_ubi.binding_point);

            openGlState.register_program("SpheresProgram", program);
        }

        view.program = program;

        view.vao.create();

        SetPositionSphereView(entity_id, "v:point").execute();
        SetRadiusSphereView(entity_id, "v:radius").execute();
        SetColorSphereView(entity_id, "v:color").execute();

        auto v_indices = vertices->get<unsigned int>("v:indices");
        if (!v_indices) {
            v_indices = vertices->add<unsigned int>("v:indices", 0);
            std::iota(v_indices.vector().begin(), v_indices.vector().end(), 0);
        }

        SetIndicesSphereView(entity_id, v_indices.vector()).execute();

        view.vao.unbind();
    }

    void SetPositionSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            SetupSphereView(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();

        view.vao.bind();

        OpenGLState openGlState(entity_id);

        auto v_positions = vertices->get<Vector<float, 3>>(property_name);
        auto b_position = openGlState.get_buffer(property_name);

        if (v_positions) {
            if (!b_position) {
                b_position = ArrayBuffer();
                b_position.create();
                openGlState.register_buffer(property_name, b_position);
            }

            b_position = openGlState.get_buffer(property_name);

            b_position.bind();
            b_position.buffer_data(v_positions.data(),
                                   num_vertices * 3 * sizeof(float),
                                   Buffer::STATIC_DRAW);

            view.position.bound_buffer_name = property_name.c_str();
            view.position.set(nullptr);
            view.position.enable();
        } else {
            Log::Warn(name + ": failed, because entity does not have " + property_name + " property.");
        }

        view.vao.unbind();

        if (v_positions) {
            b_position.unbind();
        }
    }

    void SetRadiusSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            SetupSphereView(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();

        view.vao.bind();

        OpenGLState openGlState(entity_id);

        auto v_radius = vertices->get<float>(property_name);
        auto b_radius = openGlState.get_buffer(property_name);

        if (v_radius) {
            if (!b_radius) {
                b_radius = ArrayBuffer();
                b_radius.create();
                openGlState.register_buffer(property_name, b_radius);
            }

            b_radius = openGlState.get_buffer(property_name);

            b_radius.bind();
            b_radius.buffer_data(v_radius.data(),
                                 num_vertices * 1 * sizeof(float),
                                 Buffer::STATIC_DRAW);

            if (Map(v_radius.vector()).minCoeff() < 0) {
                Log::Error(name + ": " + property_name + " has negative values which cannot be used as radius!");
            }

            view.radius.bound_buffer_name = property_name.c_str();
            view.radius.set(nullptr);
            view.radius.enable();
        } else {
            view.radius.disable();
        }
        view.radius.set_default(&view.default_radius);

        view.vao.unbind();

        if (v_radius) {
            b_radius.unbind();
        }
    }

    void SetColorSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            SetupSphereView(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();

        view.vao.bind();

        OpenGLState openGlState(entity_id);

        auto v_color = vertices->get<Vector<float, 3>>(property_name);
        auto b_color = openGlState.get_buffer(property_name);

        if (v_color) {
            if (!b_color) {
                b_color = ArrayBuffer();
                b_color.create();
                openGlState.register_buffer(property_name, b_color);
            }

            b_color = openGlState.get_buffer(property_name);

            b_color.bind();
            b_color.buffer_data(v_color.data(),
                                num_vertices * 3 * sizeof(float),
                                Buffer::STATIC_DRAW);
            view.min_color = Map(v_color.vector()).minCoeff();
            view.max_color = Map(v_color.vector()).maxCoeff();

            view.color.bound_buffer_name = property_name.c_str();
            view.color.set(nullptr);
            view.color.enable();
        } else {
            view.color.disable();
        }
        view.color.set_default(view.base_color.data());

        view.vao.unbind();

        if (v_color) {
            b_color.unbind();
        }
    }

    void SetIndicesSphereView::execute() const {
        if (indices.empty()) {
            Log::Error(name + ": failed, indices vector is empty!");
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            SetupSphereView(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();

        view.vao.bind();

        OpenGLState openGlState(entity_id);

        auto b_indices = openGlState.get_buffer("v:indices");
        if (!b_indices) {
            b_indices = ElementArrayBuffer();
            b_indices.create();
            openGlState.register_buffer("v:indices", b_indices);
        }

        b_indices = openGlState.get_buffer("v:indices");
        b_indices.bind();
        b_indices.buffer_data(indices.data(),
                              num_vertices * 1 * sizeof(unsigned int),
                              Buffer::STATIC_DRAW);
        view.vao.unbind();
        b_indices.unbind();
    }
}