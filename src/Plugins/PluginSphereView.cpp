//
// Created by alex on 02.08.24.
//

#include "PluginViewSphere.h"
#include "Engine.h"
#include "imgui.h"
#include "SphereViewGui.h"
#include "Picker.h"
#include "Camera.h"
#include "PluginGraphics.h"
#include "Transform.h"
#include "EventsCallbacks.h"
#include "Keyboard.h"
#include "OpenGLState.h"
#include "GetPrimitives.h"
#include "glad/gl.h"
#include <numeric>
#include "PropertyEigenMap.h"

namespace Bcg {
    static float global_point_size = 1.0f;

    static void on_mouse_scroll(const Events::Callback::MouseScroll &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (!keyboard.strg()) return;
        auto &picked = Engine::Context().get<Picked>();
        auto entity_id = picked.entity.id;
        if (!picked.entity.is_background && Engine::has<SphereView>(entity_id)) {
            auto &view = Engine::State().get<SphereView>(entity_id);
            view.uniform_radius = std::max<float>(1.0f, view.uniform_radius + event.yoffset);
        } else {
            global_point_size = std::max<float>(1.0f, global_point_size + event.yoffset);
            glPointSize(global_point_size);
        }
    }

    void PluginViewSphere::activate() {
        Plugin::activate();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&on_mouse_scroll>();
    }

    void PluginViewSphere::begin_frame() {}

    void PluginViewSphere::update() {}

    void PluginViewSphere::end_frame() {}

    void PluginViewSphere::deactivate() {
        Plugin::deactivate();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().disconnect<&on_mouse_scroll>();
    }

    static bool show_gui = false;

    void PluginViewSphere::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("SphereView", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginViewSphere::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("SphereView", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowSphereView(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginViewSphere::render() {
        auto rendergroup = Engine::State().view<SphereView>();
        auto &camera = Engine::Context().get<Camera>();
        auto vp = PluginGraphics::get_viewport();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<SphereView>(entity_id);
            if(view.hide) continue;

            view.vao.bind();
            view.program.use();
            view.program.set_uniform1ui("width", vp[2]);
            view.program.set_uniform1ui("height", vp[3]);
            view.program.set_uniform1i("use_uniform_radius", view.use_uniform_radius);
            view.program.set_uniform1f("uniform_radius", view.uniform_radius);
            view.program.set_uniform1f("min_color", view.min_color);
            view.program.set_uniform1f("max_color", view.max_color);
            view.program.set_uniform1i("use_uniform_color", view.use_uniform_color);
            view.program.set_uniform3fv("uniform_color", glm::value_ptr(view.uniform_color));
            view.program.set_uniform3fv("light_position", glm::value_ptr(GetViewParams(camera).eye));

            if (Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                view.program.set_uniform4fm("model", glm::value_ptr(transform.world()), false);
            } else {
                view.program.set_uniform4fm("model", glm::value_ptr(glm::mat4(1.0f)), false);
            }

            view.draw();
            view.vao.unbind();
        }
    }

    void Commands::Setup<SphereView>::execute() const {
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

        SetPositionSphereView(entity_id, "v:position").execute();
        SetRadiusSphereView(entity_id, "uniform_radius").execute();
        SetNormalSphereView(entity_id, "v:normal").execute();
        SetColorSphereView(entity_id, "uniform_color").execute();

        auto v_indices = vertices->get<unsigned int>("v:indices");
        if (!v_indices) {
            v_indices = vertices->add<unsigned int>("v:indices", 0);
            std::iota(v_indices.vector().begin(), v_indices.vector().end(), 0);
        }

        SetIndicesSphereView(entity_id, v_indices.vector()).execute();

        view.vao.unbind();
    }

    void Commands::SetPositionSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
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

    void Commands::SetRadiusSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
        size_t num_vertices = vertices->size();

        OpenGLState openGlState(entity_id);

        auto v_radius = vertices->get<float>(property_name);
        auto b_radius = openGlState.get_buffer(property_name);

        if (v_radius) {
            if (!b_radius) {
                b_radius = ArrayBuffer();
                b_radius.create();
                openGlState.register_buffer(property_name, b_radius);
            }

            view.vao.bind();
            b_radius.bind();
            b_radius.buffer_data(v_radius.data(),
                                 num_vertices * 1 * sizeof(float),
                                 Buffer::STATIC_DRAW);

            if (Map(v_radius.vector()).minCoeff() < 0) {
                Log::Error(name + ": " + property_name + " has negative values which cannot be used as radius!");
            }

            view.radius.set(nullptr);
            view.radius.enable();
            view.use_uniform_radius = false;
            view.vao.unbind();
        } else {
            view.use_uniform_radius = true;
        }
        view.radius.bound_buffer_name = property_name.c_str();

        if (v_radius) {
            b_radius.unbind();
        }
    }

    void Commands::SetColorSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
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
            if(view.min_color == view.max_color){
                view.min_color = 0;
            }

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

    void Commands::SetScalarfieldSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        bool any = false;
        auto v_colorf = vertices->get<float>(property_name);
        Eigen::Vector<float, -1> t(vertices->size());
        if(v_colorf){
            t = Map(v_colorf.vector());
            any = true;
        }
        if(!any){
            auto v_colori = vertices->get<int>(property_name);
            if(v_colori){
                t = Map(v_colori.vector()).cast<float>();
                any = true;
            }
        }

        if(!any){
            auto v_colorui = vertices->get<unsigned int>(property_name);
            if(v_colorui){
                t = Map(v_colorui.vector()).cast<float>();
                any = true;
            }
        }

        if(!any){
            auto v_colorb = vertices->get<bool>(property_name);
            if(v_colorb){
                for(size_t i = 0; i < t.size(); ++i){
                    t[i] = v_colorb[i];
                }
                any = true;
            }
        }

        if(any){
            auto v_colorf3 = vertices->get_or_add<Vector<float, 3>>(property_name + "Color3d");
            t = (t.array() - t.minCoeff()) / (t.maxCoeff() - t.minCoeff());
            Map(v_colorf3.vector()) = t * Eigen::Vector<float, 3>::Unit(0).transpose() + (1.0f - t.array()).matrix() * Eigen::Vector<float, 3>::Unit(1).transpose();
            SetColorSphereView(entity_id, property_name + "Color3d").execute();
        }
    }


    void Commands::SetNormalSphereView::execute() const {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
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

    void Commands::SetIndicesSphereView::execute() const {
        if (indices.empty()) {
            Log::Error(name + ": failed, indices vector is empty!");
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Setup<SphereView>(entity_id).execute();
        }

        auto &view = Engine::require<SphereView>(entity_id);
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