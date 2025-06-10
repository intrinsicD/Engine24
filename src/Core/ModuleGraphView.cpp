//
// Created by alex on 06.06.25.
//

#include "ModuleGraphView.h"
#include "imgui.h"
#include "Picker.h"
#include "Engine.h"
#include "GetPrimitives.h"
#include "CameraUtils.h"
#include "ModuleTransform.h"
#include "ModuleGraphics.h"
#include "OpenGLState.h"
#include "PropertyEigenMap.h"
#include "GraphEdges.h"

namespace Bcg {
    void ModuleGraphView::activate() {
        if (base_activate()) {

        }
    }

    void ModuleGraphView::deactivate() {
        if (base_deactivate()) {

        }
    }

    static bool gui_enabled = false;

    void ModuleGraphView::render_menu() {
        if (ImGui::BeginMenu("Rendering")) {
            ImGui::MenuItem("GraphView", nullptr, &gui_enabled);
            ImGui::EndMenu();
        }
    }

    void ModuleGraphView::render_gui() {
        if (gui_enabled) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("GraphView", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void ModuleGraphView::render() {
        auto rendergroup = Engine::State().view<GraphView>();
        auto &camera = Engine::Context().get<Camera>();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<GraphView>(entity_id);
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

            ModuleGraphics::draw_lines(view.num_indices);
        }
        ModuleGraphics::unbind_vao();
    }

    void ModuleGraphView::show_gui(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) return;
        if (!Engine::has<GraphView>(entity_id)) return;
        show_gui(entity_id, Engine::State().get<GraphView>(entity_id));
    }

    void ModuleGraphView::show_gui(entt::entity entity_id, GraphView &view) {
        if (!Engine::valid(entity_id)) {
            return;
        }

        ImGui::PushID("graph_view");
        auto *vertices = GetPrimitives(entity_id).vertices();
        ImGui::Checkbox("hide", &view.hide);
        if (vertices) {
            auto properties_3d = vertices->properties({3});

            static std::pair<int, std::string> curr_pos = {-1, view.position.bound_buffer_name};
            if (curr_pos.first == -1) {
                curr_pos.first = Gui::FindIndex(properties_3d, view.position.bound_buffer_name);
                if (curr_pos.first == -1) {
                    curr_pos.first = 0;
                }
            }
            if (Gui::Combo(view.position.shader_name.c_str(), curr_pos, properties_3d)) {
                set_positions(entity_id, properties_3d[curr_pos.first]);
            }
        }

        auto *edges = GetPrimitives(entity_id).edges();
        if (edges) {
            auto properties_colors = edges->properties({1, 3});
            properties_colors.emplace_back("uniform_color");
            static std::pair<int, std::string> curr_color = {-1, view.color.bound_buffer_name};

            if (curr_color.first == -1) {
                curr_color.first = Gui::FindIndex(properties_colors, view.color.bound_buffer_name);
                if (curr_color.first == -1) {
                    curr_color.first = 0;
                }
            }

            if (Gui::Combo(view.color.shader_name.c_str(), curr_color, properties_colors)) {
                auto *p_array = edges->get_base(properties_colors[curr_color.first]);
                if (p_array && p_array->dims() == 1) {
                    set_scalarfield(entity_id, properties_colors[curr_color.first]);
                } else {
                    set_colors(entity_id, properties_colors[curr_color.first]);
                }
            }

            if (view.use_uniform_color) {
                if (ImGui::ColorEdit3("##uniform_color_graph_view", glm::value_ptr(view.uniform_color))) {
                    set_uniform_color(entity_id, view.uniform_color);
                }
            } else {
                ImGui::InputFloat("min_color", &view.min_color);
                ImGui::InputFloat("max_color", &view.max_color);
            }

        }
        ImGui::PopID();
    }

    void ModuleGraphView::setup(entt::entity entity_id) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        auto *edges = GetPrimitives(entity_id).edges();
        if (!vertices) return;

        auto &view = Engine::require<GraphView>(entity_id);
        size_t num_edges = edges->size();
        view.num_indices = num_edges * 2;

        OpenGLState openGlState(entity_id);

        auto program = openGlState.get_program("GraphProgram");
        if (!program) {
            program.create_from_files("../Shaders/glsl/graph_vs.glsl",
                                      "../Shaders/glsl/graph_fs.glsl");

            auto &camera_ubi = Engine::Context().get<CameraUniformBuffer>();
            view.program.bind_uniform_block("Camera", camera_ubi.binding_point);

            openGlState.register_program("GraphProgram", program);
        }

        view.program = program;

        view.vao.create();
        auto positions = vertices->get<Vector<float, 3>>("v:position");
        set_positions(entity_id, positions.name());
        auto e_colors = edges->get<Vector<float, 3>>("v:color");
        if (e_colors) {
            set_colors(entity_id, e_colors.name());
        } else {
            set_uniform_color(entity_id, view.uniform_color);
        }

        auto halfedges = GetPrimitives(entity_id).halfedges();

        auto graph = GraphInterface(*vertices, *halfedges, *edges);
        auto e_indices = GraphGetEdges(graph);
        set_edges(entity_id, e_indices.vector());

        view.vao.unbind();
    }

    void ModuleGraphView::cleanup(entt::entity entity_id) {
        auto &view = Engine::State().get<GraphView>(entity_id);

        OpenGLState openGlState(entity_id);
        openGlState.clear();

        view.vao.destroy();
        view.program.destroy();

        Engine::State().remove<GraphView>(entity_id);
    }

    void ModuleGraphView::set_positions(entt::entity entity_id, const std::string &property_name) {
        auto *vertices = GetPrimitives(entity_id).vertices();
        if (!vertices) return;

        if (!Engine::has<GraphView>(entity_id)) {
            Log::Error("GraphView::set_positions: failed, because entity does not have GraphView component.");
            return;
        }

        auto &view = Engine::require<GraphView>(entity_id);
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
            Log::Warn("GraphView::set_postitions: failed, because entity does not have {} property.", property_name);
        }
    }

    void ModuleGraphView::set_colors(entt::entity entity_id, const std::string &property_name) {
        auto *edges = GetPrimitives(entity_id).edges();
        if (!edges) return;

        if (!Engine::has<GraphView>(entity_id)) {
            Log::Error("GraphView::set_colors: failed, because entity does not have GraphView component.");
            return;
        }

        auto &view = Engine::require<GraphView>(entity_id);
        size_t num_edges = edges->size();

        OpenGLState openGlState(entity_id);

        auto e_colors = edges->get<Vector<float, 3>>(property_name);
        auto b_edge_color = openGlState.get_buffer("edge_color");

        if (e_colors) {
            if (!b_edge_color) {
                b_edge_color = ShaderStorageBuffer();
                b_edge_color.create();
                openGlState.register_buffer(property_name, b_edge_color);
            }

            view.vao.bind();
            b_edge_color.bind();
            Eigen::Matrix<float, -1, 4> e_colors_4d(e_colors.vector().size(), 4);
            e_colors_4d.leftCols<3>() = Map(e_colors.vector());
            b_edge_color.buffer_data(e_colors_4d.data(),
                                     num_edges * 4 * sizeof(float),
                                     Buffer::STATIC_DRAW);

            view.color.bound_buffer_name = property_name.c_str();
            view.color.set(nullptr);
            view.color.enable();
            view.vao.unbind();
            b_edge_color.unbind();
        } else {
            Log::Warn("GraphView::set_colors: failed, because entity does not have {} property.", property_name);
        }
    }

    void ModuleGraphView::set_scalarfield(entt::entity entity_id, const std::string &property_name) {
        auto *edges = GetPrimitives(entity_id).edges();
        if (!edges) return;

        if (!Engine::has<GraphView>(entity_id)) {
            Log::Error("GraphView::set_scalarfield: failed, because entity does not have GraphView component.");
            return;
        }

        auto &view = Engine::require<GraphView>(entity_id);
        size_t num_edges = edges->size();

        OpenGLState openGlState(entity_id);

        auto e_scalarfield = edges->get<Vector<float, 3>>(property_name);
        auto b_edge_scalarfield = openGlState.get_buffer("edge_scalarfield");

        if (e_scalarfield) {
            if (!b_edge_scalarfield) {
                b_edge_scalarfield = ShaderStorageBuffer();
                b_edge_scalarfield.create();
                openGlState.register_buffer(property_name, b_edge_scalarfield);
            }

            view.vao.bind();
            b_edge_scalarfield.bind();
            b_edge_scalarfield.buffer_data(e_scalarfield.data(),
                                           num_edges * sizeof(float),
                                           Buffer::STATIC_DRAW);

            view.scalarfield.bound_buffer_name = property_name.c_str();
            view.scalarfield.set(nullptr);
            view.scalarfield.enable();
            view.vao.unbind();
            b_edge_scalarfield.unbind();
        } else {
            Log::Warn("GraphView::set_scalarfield: failed, because entity does not have {} property.", property_name);
        }
    }

    void ModuleGraphView::set_uniform_color(entt::entity entity_id, const Vector<float, 3> &uniform_color) {
        if (!Engine::has<GraphView>(entity_id)) {
            Log::Error("GraphView::set_uniform_color: failed, because entity does not have GraphView component.");
            return;
        }

        auto &view = Engine::require<GraphView>(entity_id);
        view.uniform_color = uniform_color;

        view.vao.bind();
        view.color.bound_buffer_name = "uniform_color";
        view.color.disable();
        view.color.set_default(glm::value_ptr(view.uniform_color));
        view.use_uniform_color = true;

        view.vao.unbind();
    }

    void ModuleGraphView::set_edges(entt::entity entity_id, const std::vector<Vector<unsigned int, 2>> &edges_indices) {
        if (edges_indices.empty()) {
            Log::Error("ModuleGraphView::set_edges: failed, edges_indices vector is empty!");
            return;
        }
        if (!Engine::has<GraphView>(entity_id)) {
            Log::Error("ModuleGraphView::set_edges: failed, because entity does not have GraphView component.");
        }

        auto &view = Engine::require<GraphView>(entity_id);
        size_t num_edges = edges_indices.size();

        OpenGLState openGlState(entity_id);
        std::string buffer_name = "e:indices";

        auto b_indices = openGlState.get_buffer(buffer_name);
        if (!b_indices) {
            b_indices = ElementArrayBuffer();
            b_indices.create();
            openGlState.register_buffer(buffer_name, b_indices);
        }

        b_indices = openGlState.get_buffer(buffer_name);

        view.vao.bind();
        b_indices.bind();
        b_indices.buffer_data(edges_indices.data(),
                              num_edges * 2 * sizeof(unsigned int),
                              Buffer::STATIC_DRAW);
        view.vao.unbind();
        b_indices.unbind();
    }
}