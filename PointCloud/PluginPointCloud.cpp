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
#include "PointCloudCommands.h"
#include "EntityCommands.h"
#include "Picker.h"

namespace Bcg {
    namespace PluginPointCloudInternal {
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
    }

    PointCloud PluginPointCloud::load(const std::string &path) {
        std::string ext = path;
        ext = ext.substr(ext.find_last_of('.') + 1);
        PointCloud pc;
        if (ext == "xyz") {
            pc = load_xyz(path);
        } else if (ext == "pts") {
            pc = load_pts(path);
        } else if (ext == "csv") {
            pc = load_csv(path);
        } else {
            Log::Error("Unsupported file format: " + ext);
            return {};
        }
        auto entity_id = Engine::State().create();
        Commands::Entity::Add<PointCloud>(entity_id, pc, "PointCloud").execute();
        Commands::Points::SetupPointCloud(entity_id).execute();
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
                if (ImGui::MenuItem("Instance", nullptr, &show_pc_gui)) {

                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginPointCloud::render_gui() {
        Gui::ShowLoadPointCloud();
        if (show_pc_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("PointCloud", &show_pc_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowPointCloud(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginPointCloud::render() {

    }
}