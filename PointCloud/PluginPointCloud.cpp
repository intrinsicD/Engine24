//
// Created by alex on 30.07.24.
//


#include "PluginPointCloud.h"
#include "Logger.h"
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "PointCloudGui.h"
#include <chrono>
#include "PointCloud.h"
#include "io/io.h"
#include "io/read_xyz.h"
#include "io/read_pts.h"
#include "io/read_csv.h"
#include "Camera.h"
#include "VertexArrayObject.h"
#include "Views.h"
#include "PointCloudCommands.h"
#include "EntityCommands.h"
#include "Picker.h"
#include "Transform.h"
#include "Keyboard.h"
#include "glad/gl.h"

namespace Bcg {
    namespace PluginPointCloudInternal{
        static void on_drop_file(const Events::Callback::Drop &event) {
            PluginPointCloud plugin;
            for (int i = 0; i < event.count; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();

                PointCloud spc = PluginPointCloud::load(event.paths[i]);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> build_duration = end_time - start_time;
                Log::Info("Build Spc in " + std::to_string(build_duration.count()) + " seconds");
            }
        }

        static float point_size = 1.0f;
        static void on_mouse_scroll(const Events::Callback::MouseScroll &event) {
            auto &keyboard = Engine::Context().get<Keyboard>();
            if (!keyboard.strg()) return;
            point_size = std::max<float>(1.0f, point_size + event.yoffset);
            glPointSize(point_size);
        }
    }
    PointCloud PluginPointCloud::load(const std::string &path) {
        std::string ext = path;
        ext = ext.substr(ext.find_last_of('.') + 1);
        PointCloud pc;
        if (ext == "xyz") {
            pc = load_xyz(path);
        } else if (ext == "pts") {
            pc = load_pts(path);
        } else  if (ext == "csv") {
            pc = load_csv(path);
        }else{
            Log::Error("Unsupported file format: " + ext);
            return {};
        }
        auto entity_id = Engine::State().create();
        Commands::Entity::Add<PointCloud>(entity_id, pc, "PointCloud").execute();
        Commands::Points::SetupForRendering(entity_id).execute();
        return pc;
    }

    PointCloud PluginPointCloud::load_xyz(const std::string &path) {
        PointCloud pc;
        read_xyz(pc, path);
        return pc;
    }

    PointCloud PluginPointCloud::load_pts(const std::string &path) {
        PointCloud pc;
        read_pts(pc, path);
        return pc;
    }

    PointCloud PluginPointCloud::load_csv(const std::string &path) {
        PointCloud pc;
        read_csv(pc, path);
        return pc;
    }



    PluginPointCloud::PluginPointCloud() : Plugin("PluginPointCloud") {}

    void PluginPointCloud::activate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&PluginPointCloudInternal::on_drop_file>();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&PluginPointCloudInternal::on_mouse_scroll>();
        Plugin::activate();
    }

    void PluginPointCloud::begin_frame() {

    }

    void PluginPointCloud::update() {

    }

    void PluginPointCloud::end_frame() {

    }

    void PluginPointCloud::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&PluginPointCloudInternal::on_drop_file>();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().disconnect<&PluginPointCloudInternal::on_mouse_scroll>();
        Plugin::deactivate();
    }

    static bool show_pc_gui = false;

    void PluginPointCloud::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            if (ImGui::BeginMenu("PointCloud")) {
                if (ImGui::MenuItem("Load PointCloud")) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    config.path = "/home/alex/Dropbox/Work/Datasets";
                    ImGuiFileDialog::Instance()->OpenDialog("Load PointCloud", "Choose File", ".xyz,.pts,.csv",
                                                            config);
                }
                if(ImGui::MenuItem("Instance", nullptr, &show_pc_gui)){

                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginPointCloud::render_gui() {
        Gui::ShowLoadPointCloud();
        if(show_pc_gui){
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("PointCloud", &show_pc_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowPointCloud(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginPointCloud::render() {
        auto pc_view = Engine::State().view<PointCloudView>();
        auto &camera = Engine::Context().get<Camera>();
        auto lightDirection = (camera.v_params.center - camera.v_params.eye).normalized();

        for (auto entity_id: pc_view) {
            auto &pcw = Engine::State().get<PointCloudView>(entity_id);

            pcw.vao.bind();
            pcw.program.use();
            pcw.program.set_uniform3fv("lightDir", lightDirection.data());

            if(Engine::has<Transform>(entity_id)){
                auto &transform = Engine::State().get<Transform>(entity_id);
                pcw.program.set_uniform4fm("model", transform.data(), false);
            }else{
                pcw.program.set_uniform4fm("model", Transform().data(), false);
            }

            pcw.draw();
        }
    }
}