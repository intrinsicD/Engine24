//
// Created by alex on 04.08.24.
//

#include "ModuleMeshView.h"
#include "ModuleMesh.h"
#include "Engine.h"
#include "imgui.h"
#include "Picker.h"
#include "CameraUtils.h"
#include "EventsEntity.h"
#include "ModuleTransform.h"
#include "GetPrimitives.h"
#include "OpenGLState.h"
#include "SurfaceMeshTriangles.h"
#include "PropertyEigenMap.h"
#include "GuiUtils.h"

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

    void ModuleMeshView::activate() {
        if (base_activate()) {
            Engine::Dispatcher().sink<Events::Entity::Destroy>().connect<&on_destroy>();
        }
    }

    void ModuleMeshView::begin_frame() {}

    void ModuleMeshView::update() {}

    void ModuleMeshView::end_frame() {}

    void ModuleMeshView::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::Entity::Destroy>().disconnect<&on_destroy>();
        }
    }

    static bool gui_enabled = false;

    void ModuleMeshView::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("MeshView", nullptr, &gui_enabled);
            ImGui::EndMenu();
        }
    }

    void ModuleMeshView::render_gui() {
        if (gui_enabled) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("MeshView", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void ModuleMeshView::show_gui(entt::entity entity_id, MeshView &view){
        if(!Engine::valid(entity_id)) {
            return;
        }

        ImGui::PushID("mesh_view");
        auto *vertices = GetPrimitives(entity_id).vertices();
        ImGui::Checkbox("hide", &view.hide);
        if (vertices) {
            auto properties_3d = vertices->properties({3});

            static std::pair<int, std::string> curr_pos = {-1, view.position.bound_buffer_name};
            if(curr_pos.first == -1){
                curr_pos.first = Gui::FindIndex(properties_3d, view.position.bound_buffer_name);
                if(curr_pos.first == -1){
                    curr_pos.first = 0;
                }
            }
            if (Gui::Combo(view.position.shader_name.c_str(), curr_pos, properties_3d)) {
                set_positions(entity_id, properties_3d[curr_pos.first]);
            }

            static std::pair<int, std::string> curr_normal = {-1, view.normal.bound_buffer_name};
            if(curr_normal.first == -1){
                curr_normal.first = Gui::FindIndex(properties_3d, view.normal.bound_buffer_name);
                if(curr_normal.first == -1){
                    curr_normal.first = 0;
                }
            }
            if (Gui::Combo(view.normal.shader_name.c_str(), curr_normal, properties_3d)) {
                set_normals(entity_id, properties_3d[curr_normal.first]);
            }

            {
                auto properties_colors = vertices->properties({1, 3});
                properties_colors.emplace_back("uniform_color");
                static std::pair<int, std::string> curr_color = {-1, view.color.bound_buffer_name};

                if(curr_color.first == -1){
                    curr_color.first = Gui::FindIndex(properties_colors, view.color.bound_buffer_name);
                    if(curr_color.first == -1){
                        curr_color.first = 0;
                    }
                }

                if (Gui::Combo(view.color.shader_name.c_str(), curr_color, properties_colors)) {
                    auto *p_array = vertices->get_base(properties_colors[curr_color.first]);
                    if(p_array && p_array->dims() == 1){
                        set_scalarfield(entity_id, properties_colors[curr_color.first]);
                    }else{
                        set_colors(entity_id, properties_colors[curr_color.first]);
                    }
                }

                if (view.use_uniform_color) {
                    if (ImGui::ColorEdit3("##uniform_color_mesh_view", glm::value_ptr(view.uniform_color))) {
                        set_uniform_color(entity_id, view.uniform_color);
                    }
                } else {
                    ImGui::InputFloat("min_color", &view.min_color);
                    ImGui::InputFloat("max_color", &view.max_color);
                }
            }
        }
        ImGui::PopID();
    }

    void ModuleMeshView::show_gui(entt::entity entity_id){
        if(!Engine::valid(entity_id) || !Engine::has<MeshView>(entity_id)) {
            return;
        }
        show_gui(entity_id, Engine::State().get<MeshView>(entity_id));
    }

    void ModuleMeshView::render() {
        auto rendergroup = Engine::State().view<MeshView>();
        auto &camera = Engine::Context().get<Camera>();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<MeshView>(entity_id);
            if (view.hide) continue;

            view.vao.bind();
            view.program.use();
            view.program.set_uniform3fv("light_position", glm::value_ptr(GetViewParams(camera).eye));
            view.program.set_uniform1f("min_color", view.min_color);
            view.program.set_uniform1f("max_color", view.max_color);
            view.program.set_uniform1i("use_uniform_color", view.use_uniform_color);
            view.program.set_uniform3fv("uniform_color", glm::value_ptr(view.uniform_color));

            if (Engine::has<TransformHandle>(entity_id)) {
                auto h_transform = Engine::State().get<TransformHandle>(entity_id);
                view.program.set_uniform4fm("model", glm::value_ptr(h_transform->world()), false);
            } else {
                view.program.set_uniform4fm("model", glm::value_ptr(glm::mat4(1.0f)), false);
            }

            view.draw();
            view.vao.unbind();
        }
    }

    void ModuleMeshView::setup(entt::entity entity_id) {
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
        auto positions = vertices->get<Vector<float, 3>>("v:position");
        set_positions(entity_id, positions.name());
        auto normals = vertices->get<Vector<float, 3>>("v:normal");
        set_normals(entity_id, normals.name());
        auto colors = vertices->get<Vector<float, 3>>("v:color");
        if (colors) {
            set_colors(entity_id, colors.name());
        } else {
            set_uniform_color(entity_id, view.uniform_color);
        }

        auto h_mesh = Engine::State().get<MeshHandle>(entity_id);
        auto f_triangles = SurfaceMeshTriangles(h_mesh);
        set_triangles(entity_id, f_triangles.vector());

        view.vao.unbind();
    }

    void ModuleMeshView::cleanup(entt::entity entity_id) {
        auto &view = Engine::State().get<MeshView>(entity_id);

        OpenGLState openGlState(entity_id);
        openGlState.clear();

        view.vao.destroy();
        view.program.destroy();

        Engine::State().remove<MeshView>(entity_id);
    }

    void ModuleMeshView::set_positions(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Log::Error("MeshView::set_positions: failed, because entity does not have MeshView component.");
            return;
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
            Log::Warn("MeshView::set_postitions: failed, because entity does not have {} property.", property_name);
        }
    }

    void ModuleMeshView::set_normals(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Log::Error("MeshView::set_normals: failed, because entity does not have MeshView component.");
            return;
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
            Log::Warn("MeshView::set_normals: failed, because entity does not have {} property.", property_name);
        }
    }

    void ModuleMeshView::set_colors(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Log::Error("MeshView::set_colors: failed, because entity does not have MeshView component.");
            return;
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

    void ModuleMeshView::set_uniform_color(entt::entity entity_id, const Vector<float, 3> &data) {
         if (!Engine::has<MeshView>(entity_id)) {
            Log::Error("MeshView::set_uniform_color: failed, because entity does not have MeshView component.");
            return;
        }

        auto &view = Engine::require<MeshView>(entity_id);

        view.vao.bind();
        view.color.bound_buffer_name = "uniform_color";
        view.color.disable();
        view.color.set_default(glm::value_ptr(view.uniform_color));
        view.use_uniform_color = true;

        view.vao.unbind();
    }

    void ModuleMeshView::set_scalarfield(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<MeshView>(entity_id)) {
            Log::Error("MeshView::set_scalarfield: failed, because entity does not have MeshView component.");
            return;
        }

        auto v_colorf = vertices->get<float>(property_name);
        Eigen::Vector<float, -1> t(vertices->size());
        if (v_colorf) {
            t = Map(v_colorf.vector());
        }
        auto v_colord = vertices->get<double>(property_name);
        if (v_colord) {
            t = Map(v_colord.vector()).cast<float>();
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

        auto color_property_name = property_name + "Color3d";
        auto v_colorf3 = vertices->get_or_add<Vector<float, 3>>(color_property_name);
        t = (t.array() - t.minCoeff()) / (t.maxCoeff() - t.minCoeff());
        //Todo : use colormap here or directly in the shader
        Map(v_colorf3.vector()).transpose() = t * Eigen::Vector<float, 3>::Unit(0).transpose() +
                                              (1.0f - t.array()).matrix() *
                                              Eigen::Vector<float, 3>::Unit(1).transpose();
        set_colors(entity_id, color_property_name);
    }

    void ModuleMeshView::set_triangles(entt::entity entity_id, std::vector<Vector<unsigned int, 3>> &tris) {
        if (tris.empty()) {
            Log::Error("MeshView::set_triangles: failed, tris vector is empty!");
            return;
        }
        if (!Engine::has<MeshView>(entity_id)) {
            Log::Error("MeshView::set_triangles: failed, because entity does not have MeshView component.");
        }

        auto *faces = GetPrimitives(entity_id).faces();
        if (!faces) return;


        auto &view = Engine::require<MeshView>(entity_id);
        size_t num_faces = faces->size();

        OpenGLState openGlState(entity_id);
        std::string buffer_name = "f:indices";

        auto b_indices = openGlState.get_buffer(buffer_name);
        if (!b_indices) {
            b_indices = ElementArrayBuffer();
            b_indices.create();
            openGlState.register_buffer(buffer_name, b_indices);
        }

        b_indices = openGlState.get_buffer(buffer_name);

        view.vao.bind();
        b_indices.bind();
        b_indices.buffer_data(tris.data(),
                              num_faces * 3 * sizeof(unsigned int),
                              Buffer::STATIC_DRAW);
        view.vao.unbind();
        b_indices.unbind();
    }
}