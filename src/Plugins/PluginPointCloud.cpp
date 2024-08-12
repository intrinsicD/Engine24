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
#include "PointCloudCommands.h"
#include "Picker.h"

namespace Bcg {
    namespace PluginPointCloudInternal {
        static void on_drop_file(const Events::Callback::Drop &event) {
            PluginPointCloud plugin;
            for (int i = 0; i < event.count; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();

                PointCloud pc = PluginPointCloud::load(event.paths[i]);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> build_duration = end_time - start_time;
                Log::Info("Build PointCloud in " + std::to_string(build_duration.count()) + " seconds");
            }
        }
    }

    PointCloud PluginPointCloud::load(const std::string &filepath) {
        auto entity_id = Engine::State().create();
        Commands::Points::LoadPointCloud(entity_id, filepath).execute();
        Commands::Points::SetupPointCloud(entity_id).execute();
        return Engine::require<PointCloud>(entity_id);
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