//
// Created by alex on 04.08.24.
//


#include "PluginMeshView.h"
#include "Engine.h"
#include "imgui.h"
#include "MeshViewGui.h"
#include "Picker.h"
#include "Camera.h"
#include "EventsEntity.h"
#include "Transform.h"
#include "GetPrimitives.h"
#include "OpenGLState.h"
#include "SurfaceMeshTriangles.h"
#include "PropertyEigenMap.h"

namespace Bcg {

    void on_destroy(const Events::Entity::Destroy &event) {
        auto entity_id = event.entity_id;

        if (!Engine::valid(entity_id)) {
            return;
        }

        if (!Engine::has<MeshView>(entity_id)) {
            return;
        }

        auto &view = Engine::require<MeshView>(entity_id);
        OpenGLState openGlState(entity_id);
        openGlState.clear();
    }

    void PluginMeshView::activate() {
        Engine::Dispatcher().sink<Events::Entity::Destroy>().connect<&on_destroy>();
        Plugin::activate();
    }

    void PluginMeshView::begin_frame() {}

    void PluginMeshView::update() {}

    void PluginMeshView::end_frame() {}

    void PluginMeshView::deactivate() {
        Engine::Dispatcher().sink<Events::Entity::Destroy>().disconnect<&on_destroy>();
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginMeshView::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("MeshView", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginMeshView::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("MeshView", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowMeshView(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginMeshView::render() {
        auto rendergroup = Engine::State().view<MeshView>();
        auto &camera = Engine::Context().get<Camera>();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<MeshView>(entity_id);
            if (view.hide) continue;

            view.vao.bind();
            view.program.use();
            view.program.set_uniform3fv("light_position", camera.v_params.eye.data());
            view.program.set_uniform1f("min_color", view.min_color);
            view.program.set_uniform1f("max_color", view.max_color);
            view.program.set_uniform1i("use_uniform_color", view.use_uniform_color);
            view.program.set_uniform3fv("uniform_color", view.uniform_color.data());

            if (Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                view.program.set_uniform4fm("model", transform.data(), false);
            } else {
                view.program.set_uniform4fm("model", Transform().data(), false);
            }

            view.draw();
            view.vao.unbind();
        }
    }

    void Commands::Setup<MeshView>::execute() const {
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

        SetPositionMeshView(entity_id, "v:position").execute();
        SetNormalMeshView(entity_id, "v:normal").execute();
        SetColorMeshView(entity_id, "uniform_color").execute();

        auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
        auto f_triangles = SurfaceMeshTriangles(mesh);
        SetTrianglesMeshView(entity_id, f_triangles.vector()).execute();

        view.vao.unbind();
    }

    void Commands::Cleanup<MeshView>::execute() const {
        auto &view = Engine::State().get<MeshView>(entity_id);
        Log::TODO("Cleanup<MeshView> not implemented!");
    }

    void Commands::SetPositionMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Setup<MeshView>(entity_id).execute();
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

    void Commands::SetNormalMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Setup<MeshView>(entity_id).execute();
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

    void Commands::SetColorMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Setup<MeshView>(entity_id).execute();
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

    void Commands::SetScalarfieldMeshView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Setup<MeshView>(entity_id).execute();
        }

        auto v_colorf = vertices->get<float>(property_name);
        Vector<float, -1> t(vertices->size());
        if (v_colorf) {
            t = Map(v_colorf.vector());
        }
        auto v_colori = vertices->get<int>(property_name);
        if (v_colori) {
            t = Map(v_colori.vector()).cast<float>();
        }
        auto v_colorui = vertices->get<unsigned int>(property_name);
        if (v_colorui) {
            t = Map(v_colorui.vector()).cast<float>();
        }
        auto v_colorb = vertices->get<bool>(property_name);
        if (v_colorb) {
            for (size_t i = 0; i < t.size(); ++i) {
                t[i] = v_colorb[i];
            }
        }

        auto v_colorf3 = vertices->get_or_add<Vector<float, 3>>(property_name + "Color3d");
        t = (t.array() - t.minCoeff()) / (t.maxCoeff() - t.minCoeff());
        Map(v_colorf3.vector()).transpose() = t * Vector<float, 3>::Unit(0).transpose() +
                                              (1.0f - t.array()).matrix() * Vector<float, 3>::Unit(1).transpose();
        SetColorMeshView(entity_id, property_name + "Color3d").execute();
    }

    void Commands::SetTrianglesMeshView::execute() const {
        if (tris.empty()) {
            Log::Error(name + ": failed, tris vector is empty!");
            return;
        }

        auto *faces = GetPrimitives(entity_id).faces();
        if (!faces) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Setup<MeshView>(entity_id).execute();
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