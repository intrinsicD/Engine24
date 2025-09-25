//
// Created by alex on 05.08.24.
//

#include "PluginViewVectorfields.h"
#include "ModuleAABB.h"
#include "Engine.h"
#include "imgui.h"
#include "VectorfieldViewGui.h"
#include "Picker.h"
#include "Camera.h"
#include "ModuleGraphics.h"
#include "../../include/Core/Transform/WorldTransformComponent.h"
#include "GetPrimitives.h"
#include "PropertyEigenMap.h"
#include "OpenGLState.h"
#include "AABBComponents.h"

#include <numeric>


namespace Bcg {
    void PluginViewVectorfields::activate() {
        if (base_activate()) {

        }
    }

    void PluginViewVectorfields::begin_frame() {}

    void PluginViewVectorfields::update() {}

    void PluginViewVectorfields::end_frame() {}

    void PluginViewVectorfields::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool show_gui = false;

    void PluginViewVectorfields::render_menu() {
        if (ImGui::BeginMenu("Rendering")) {
            if(ImGui::BeginMenu("Views")){
                ImGui::MenuItem("Vectorfield", nullptr, &show_gui);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginViewVectorfields::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("Views - Vectorfield", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowVectorfieldViews(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginViewVectorfields::render() {
        auto rendergroup = Engine::State().view<VectorfieldViews>();
        for (auto entity_id: rendergroup) {
            auto &views = Engine::State().get<VectorfieldViews>(entity_id);
            if (views.hide) continue;
            for (auto &item: views.vectorfields) {
                auto &view = item.second;

                if (view.hide) continue;

                view.vao.bind();
                view.program.bind();
                view.program.set_uniform1i("use_uniform_length", view.use_uniform_length);
                view.program.set_uniform1f("uniform_length", view.uniform_length);
                view.program.set_uniform1f("min_color", view.min_color);
                view.program.set_uniform1f("max_color", view.max_color);
                view.program.set_uniform1i("use_uniform_color", view.use_uniform_color);
                view.program.set_uniform3fv("uniform_color", glm::value_ptr(view.uniform_color));

                if (Engine::has<WorldTransformComponent>(entity_id)) {
                    auto &transform = Engine::State().get<WorldTransformComponent>(entity_id);
                    view.program.set_uniform4fm("model", glm::value_ptr(transform.world_transform), false);
                } else {
                    view.program.set_uniform4fm("model", glm::value_ptr(glm::mat4(1.0f)), false);
                }

                ModuleGraphics::draw_points(view.num_vectors);
            }
            ModuleGraphics::unbind_vao();
        }
    }

    void Commands::Setup<VectorfieldView>::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        auto &views = Engine::require<VectorfieldViews>(entity_id);
        auto &view = (views.vectorfields.emplace(vectorfield_name, VectorfieldView()).first)->second;
        view.vectorfield_name = vectorfield_name;
        size_t num_vertices = vertices->size();
        view.num_vectors = num_vertices;

        OpenGLState openGlState(entity_id);

        auto program = openGlState.get_program("VectorfieldsProgram");
        if (!program) {
            program.create_from_files("../Shaders/glsl/vector_vs.glsl",
                                      "../Shaders/glsl/vector_fs.glsl",
                                      "../Shaders/glsl/vector_gs.glsl");

            auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
            view.program.bind_uniform_block("Camera", camera_ubi.binding_point);

            openGlState.register_program("VectorfieldsProgram", program);
        }

        view.program = program;

        view.vao.create();

        SetPositionVectorfieldView(entity_id, vectorfield_name, "v:point").execute();
        SetLengthVectorfieldView(entity_id, vectorfield_name, "uniform_length").execute();
        SetVectorVectorfieldView(entity_id, vectorfield_name, vectorfield_name).execute();
        SetColorVectorfieldView(entity_id, vectorfield_name, "uniform_color").execute();

        if (Engine::has<LocalAABB>(entity_id)) {
            auto local = Engine::State().get<LocalAABB>(entity_id);
            view.uniform_length = glm::length(local.aabb.diagonal()) / 100.0f;
        } else {
            view.uniform_length = 1.0f;
        }

        auto v_indices = vertices->get<unsigned int>("v:indices");
        if (!v_indices) {
            v_indices = vertices->add<unsigned int>("v:indices", 0);
            std::iota(v_indices.vector().begin(), v_indices.vector().end(), 0);
        }

        SetIndicesVectorfieldView(entity_id, vectorfield_name, v_indices.vector()).execute();

        view.vao.unbind();
    }

    void Commands::SetPositionVectorfieldView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<VectorfieldViews>(entity_id)) {
            Setup<VectorfieldView>(entity_id, vectorfield_name).execute();
        }

        auto &views = Engine::require<VectorfieldViews>(entity_id);
        auto &view = views.vectorfields[vectorfield_name];
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

    void Commands::SetLengthVectorfieldView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<VectorfieldViews>(entity_id)) {
            Setup<VectorfieldView>(entity_id, vectorfield_name).execute();
        }

        auto &views = Engine::require<VectorfieldViews>(entity_id);
        auto &view = views.vectorfields[vectorfield_name];
        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto v_length = vertices->get<float>(property_name);
        auto b_length = openGlState.get_buffer(property_name);

        if (v_length) {
            if (!b_length) {
                b_length = ArrayBuffer();
                b_length.create();
                openGlState.register_buffer(property_name, b_length);
            }

            view.vao.bind();
            b_length.bind();
            b_length.buffer_data(v_length.data(),
                                 num_vertices * 1 * sizeof(float),
                                 Buffer::STATIC_DRAW);

            if (Map(v_length.vector()).minCoeff() < 0) {
                Log::Error(name + ": " + property_name + " has negative values which cannot be used as length!");
            }

            view.length.set(nullptr);
            view.length.enable();
            view.use_uniform_length = false;
            view.vao.unbind();
        } else {
            view.use_uniform_length = true;
        }
        view.length.bound_buffer_name = property_name.c_str();

        if (v_length) {
            b_length.unbind();
        }
    }

    void Commands::SetColorVectorfieldView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<VectorfieldViews>(entity_id)) {
            Setup<VectorfieldView>(entity_id, vectorfield_name).execute();
        }

        auto &views = Engine::require<VectorfieldViews>(entity_id);
        auto &view = views.vectorfields[vectorfield_name];

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

            view.color.set(nullptr);
            view.color.enable();
            view.use_uniform_color = false;
            view.vao.unbind();
        } else {
            view.use_uniform_color = true;
        }
        view.color.bound_buffer_name = property_name.c_str();

        if (v_color) {
            b_color.unbind();
        }
    }


    void Commands::SetVectorVectorfieldView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<VectorfieldViews>(entity_id)) {
            Setup<VectorfieldView>(entity_id, vectorfield_name).execute();
        }

        auto &views = Engine::require<VectorfieldViews>(entity_id);
        auto &view = views.vectorfields[vectorfield_name];

        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto v_vectors = vertices->get<Vector<float, 3>>(property_name);
        auto b_vectors = openGlState.get_buffer(property_name);

        if (v_vectors) {
            if (!b_vectors) {
                b_vectors = ArrayBuffer();
                b_vectors.create();
                openGlState.register_buffer(property_name, b_vectors);
            }

            view.vao.bind();
            b_vectors.bind();
            b_vectors.buffer_data(v_vectors.data(),
                                  num_vertices * 3 * sizeof(float),
                                  Buffer::STATIC_DRAW);

            view.vector.bound_buffer_name = property_name.c_str();
            view.vector.set(nullptr);
            view.vector.enable();
            view.vao.unbind();
            b_vectors.unbind();
        } else {
            Log::Warn(name + ": failed, because entity does not have " + property_name + " property.");
        }
    }

    void Commands::SetIndicesVectorfieldView::execute() const {
        if (indices.empty()) {
            Log::Error(name + ": failed, indices vector is empty!");
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<VectorfieldViews>(entity_id)) {
            Setup<VectorfieldView>(entity_id, vectorfield_name).execute();
        }

        auto &views = Engine::require<VectorfieldViews>(entity_id);
        auto &view = views.vectorfields[vectorfield_name];

        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto b_indices = openGlState.get_buffer("v:indices");
        if (!b_indices) {
            b_indices = ElementArrayBuffer();
            b_indices.create();
            openGlState.register_buffer("v:indices", b_indices);
        }

        b_indices = openGlState.get_buffer("v:indices");
        view.vao.bind();
        b_indices.bind();
        b_indices.buffer_data(indices.data(),
                              num_vertices * 1 * sizeof(unsigned int),
                              Buffer::STATIC_DRAW);
        view.vao.unbind();
        b_indices.unbind();
    }
}
