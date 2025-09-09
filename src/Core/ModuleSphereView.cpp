//
// Created by alex on 02.08.24.
//

#include "ModuleSphereView.h"
#include "Engine.h"
#include "imgui.h"
#include "Picker.h"
#include "CameraUtils.h"
#include "ModuleGraphics.h"
#include "WorldTransformComponent.h"
#include "EventsCallbacks.h"
#include "Keyboard.h"
#include "OpenGLState.h"
#include "GetPrimitives.h"
#include "glad/gl.h"
#include <numeric>
#include "PropertyEigenMap.h"

namespace Bcg {
    float global_point_size = 1.0f;

    void on_mouse_scroll(const Events::Callback::MouseScroll &event) {
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

    void ModuleSphereView::activate() {
        if (base_activate()) {
            Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&on_mouse_scroll>();
        }
    }

    void ModuleSphereView::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::Callback::MouseScroll>().disconnect<&on_mouse_scroll>();
        }
    }

    bool gui_enabled = false;

    void ModuleSphereView::render_menu() {
        if (ImGui::BeginMenu("Rendering")) {
            if(ImGui::BeginMenu("Views")) {
                ImGui::MenuItem("Sphere", nullptr, &gui_enabled);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void ModuleSphereView::render_gui() {
        if (gui_enabled) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("Views - Sphere", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void ModuleSphereView::render() {
        auto rendergroup = Engine::State().view<SphereView>();
        auto &camera = Engine::Context().get<Camera>();
        auto vp = ModuleGraphics::get_viewport();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<SphereView>(entity_id);
            if (view.hide) continue;

            view.vao.bind();
            view.program.bind();
            view.program.set_uniform1ui("width", vp[2]);
            view.program.set_uniform1ui("height", vp[3]);
            view.program.set_uniform1i("use_uniform_radius", view.use_uniform_radius);
            view.program.set_uniform1f("uniform_radius", view.uniform_radius);
            view.program.set_uniform1f("min_color", view.min_color);
            view.program.set_uniform1f("max_color", view.max_color);
            view.program.set_uniform1i("use_uniform_color", view.use_uniform_color);
            view.program.set_uniform3fv("uniform_color", glm::value_ptr(view.uniform_color));
            view.program.set_uniform3fv("light_position", glm::value_ptr(GetViewParams(camera).eye));

            if (Engine::has<WorldTransformComponent>(entity_id)) {
                auto &transform = Engine::State().get<WorldTransformComponent>(entity_id);
                view.program.set_uniform4fm("model", glm::value_ptr(transform.world_transform), false);
            } else {
                view.program.set_uniform4fm("model", glm::value_ptr(glm::mat4(1.0f)), false);
            }

            ModuleGraphics::draw_points(view.num_spheres);
        }
        ModuleGraphics::unbind_vao();
    }

    void ModuleSphereView::show_gui(entt::entity entity_id, const SphereView &view) {
        if (Engine::valid(entity_id) && Engine::has<SphereView>(entity_id)) {
            ImGui::PushID("sphere_view");
            auto &view = Engine::State().get<SphereView>(entity_id);
            auto *vertices = GetPrimitives(entity_id).vertices();
            ImGui::Checkbox("hide", &view.hide);
            if (vertices) {
                auto properties_3d = vertices->properties({3});
                std::pair<int, std::string> curr_pos = {-1, view.position.bound_buffer_name};
                if (curr_pos.first == -1) {
                    curr_pos.first = Gui::FindIndex(properties_3d, view.position.bound_buffer_name);
                    if (curr_pos.first == -1) {
                        curr_pos.first = 0;
                    }
                }
                if (Gui::Combo(view.position.shader_name.c_str(), curr_pos, properties_3d)) {
                    set_position(entity_id, properties_3d[curr_pos.first]);
                }

                std::pair<int, std::string> curr_normal = {-1, view.normal.bound_buffer_name};
                if (curr_normal.first == -1) {
                    curr_normal.first = Gui::FindIndex(properties_3d, view.normal.bound_buffer_name);
                    if (curr_normal.first == -1) {
                        curr_normal.first = 0;
                    }
                }
                if (Gui::Combo(view.normal.shader_name.c_str(), curr_normal, properties_3d)) {
                    set_normal(entity_id, properties_3d[curr_normal.first]);
                }

                {
                    auto properties_colors = vertices->properties({1, 3});
                    properties_colors.emplace_back("uniform_color");
                    std::pair<int, std::string> curr_color = {-1, view.color.bound_buffer_name};
                    if (curr_color.first == -1) {
                        curr_color.first = Gui::FindIndex(properties_colors, view.color.bound_buffer_name);
                        if (curr_color.first == -1) {
                            curr_color.first = 0;
                        }
                    }

                    if (Gui::Combo(view.color.shader_name.c_str(), curr_color, properties_colors)) {
                        auto *p_array = vertices->get_base(properties_colors[curr_color.first]);
                        if (p_array && p_array->dims() == 1) {
                            set_scalarfield(entity_id,properties_colors[curr_color.first]);
                        } else {
                            set_color(entity_id, properties_colors[curr_color.first]);
                        }
                    }

                    if (view.use_uniform_color) {
                        if(ImGui::ColorEdit3("##uniform_color_sphere_view", glm::value_ptr(view.uniform_color))){
                            set_uniform_color(entity_id, view.uniform_color);
                        }
                    } else {
                        ImGui::InputFloat("min_color", &view.min_color);
                        ImGui::InputFloat("max_color", &view.max_color);
                    }
                }

                {
                    auto properties_1d = vertices->properties({1});
                    properties_1d.emplace_back("uniform_radius");
                    std::pair<int, std::string> curr_radius = {-1, view.radius.bound_buffer_name};
                    if (curr_radius.first == -1) {
                        curr_radius.first = Gui::FindIndex(properties_1d, view.radius.bound_buffer_name);
                        if (curr_radius.first == -1) {
                            curr_radius.first = 0;
                        }
                    }

                    if (Gui::Combo(view.radius.shader_name.c_str(), curr_radius, properties_1d)) {
                        set_radius(entity_id, properties_1d[curr_radius.first]);
                    }

                    if (view.use_uniform_radius) {
                        if(ImGui::InputFloat("##uniform_radius", &view.uniform_radius)){
                            set_uniform_radius(entity_id, view.uniform_radius);
                        }
                    }
                }
            }
            ImGui::Text("num_spheres: %d", view.num_spheres);
            ImGui::PopID();
        }
    }

    void ModuleSphereView::show_gui(entt::entity entity_id) {
        if (!Engine::valid(entity_id) || !Engine::has<SphereView>(entity_id)) return;
        show_gui(entity_id, Engine::State().get<SphereView>(entity_id));
    }

    void ModuleSphereView::setup(entt::entity entity_id) {
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

        set_position(entity_id, "v:point");
        set_uniform_radius(entity_id, 1.0f);
        set_normal(entity_id, "v:normal");
        set_uniform_color(entity_id, view.uniform_color);

        auto v_indices = vertices->get<unsigned int>("v:indices");
        if (!v_indices) {
            v_indices = vertices->add<unsigned int>("v:indices", 0);
            std::iota(v_indices.vector().begin(), v_indices.vector().end(), 0);
        }

        set_indices(entity_id, v_indices.vector());

        view.vao.unbind();
    }

    static std::string s_name = "ModuleSphereView";

    void ModuleSphereView::set_position(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_position: failed, because entity does not have SphereView component.", s_name);
            return;
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

            view.position.bound_buffer_name = property_name;
            view.position.set(nullptr);
            view.position.enable();
            view.vao.unbind();
            b_position.unbind();
        } else {
            Log::Warn("{}: failed, because entity does not have {} property.", s_name, property_name);
        }
    }

    void ModuleSphereView::set_normal(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_normal: failed, because entity does not have SphereView component.", s_name);
            return;
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
            Log::Warn("{}::set_normal: failed, because entity does not have {} property.", s_name, property_name);
        }
    }

    void ModuleSphereView::set_color(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_color: failed, because entity does not have SphereView component.", s_name);
            return;
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

            view.color.bound_buffer_name = property_name.c_str();
            view.color.set(nullptr);
            view.color.enable();
            view.use_uniform_color = false;
            view.vao.unbind();
            b_color.unbind();
        } else {
            set_uniform_color(entity_id, view.uniform_color);
        }
    }

    void ModuleSphereView::set_scalarfield(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_scalarfield: failed, because entity does not have SphereView component.", s_name);
            return;
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
            std::string property_name_3d = property_name + "Color3d";
            auto v_colorf3 = vertices->get_or_add<Vector<float, 3>>(property_name_3d);

            // Ensure the destination vector has the correct size before creating a map
            v_colorf3.vector().resize(vertices->size());

            // Create a map to the destination memory. It's now safe because the size matches.
            // Note that Eigen matrices are column-major by default.
            // We will treat your data as a block of XYZXYZ...
            // This requires a map of size 3xN and then transposing it for easier assignments.

            auto color_map = Map(v_colorf3.vector()).transpose(); // Now color_map is 3 x N

            // Normalize the scalar field t to the range [0, 1]
            float min_val = t.minCoeff();
            float max_val = t.maxCoeff();
            float range = max_val - min_val;

            // Avoid division by zero if all values are the same
            if (range > 0) {
                t = (t.array() - min_val) / range;
            } else {
                t.setZero(); // Or set to 0.5, or whatever makes sense
            }

            // --- FIX: Assign colors column by column ---
            // Interpolate between Green (Color1) and Red (Color2)
            // Red component
            color_map.col(0) = t;
            // Green component
            color_map.col(1) = 1.0f - t.array();
            // Blue component
            color_map.col(2).setZero();

            set_color(entity_id, property_name_3d);
        }
    }

    void ModuleSphereView::set_radius(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_radius: failed, because entity does not have SphereView component.", s_name);
            return;
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
                Log::Warn("{}::set_radius: {} has negative values which cannot be used as radius!", s_name, property_name);
            }

            view.radius.bound_buffer_name = property_name;
            view.radius.set(nullptr);
            view.radius.enable();
            view.use_uniform_radius = false;
            view.vao.unbind();
            b_radius.unbind();
        } else {
            set_uniform_radius(entity_id, view.uniform_radius);
        }
    }

    void ModuleSphereView::set_uniform_radius(entt::entity entity_id, float radius) {
        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_uniform_radius: failed, because entity does not have SphereView component.", s_name);
            return;
        }

        auto &view = Engine::require<SphereView>(entity_id);

        view.vao.bind();
        view.radius.bound_buffer_name = "uniform_radius";
        view.radius.disable();
        view.radius.set_default(&view.uniform_radius);
        view.use_uniform_radius = true;

        view.vao.unbind();
    }

    void ModuleSphereView::set_uniform_color(entt::entity entity_id, const Vector<float, 3> &color) {
        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_uniform_color: failed, because entity does not have SphereView component.", s_name);
            return;
        }

        auto &view = Engine::require<SphereView>(entity_id);

        view.vao.bind();
        view.color.bound_buffer_name = "uniform_color";
        view.color.disable();
        view.color.set_default(glm::value_ptr(view.uniform_color));
        view.use_uniform_color = true;

        view.vao.unbind();
    }

    void ModuleSphereView::set_indices(entt::entity entity_id, const std::vector<unsigned int> &indices) {
        if (indices.empty()) {
            Log::Error("{}::set_indices: failed, indices vector is empty!", s_name);
            return;
        }

        if (!Engine::has<SphereView>(entity_id)) {
            Log::Error("{}::set_indices: failed, because entity does not have SphereView component.", s_name);
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;


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