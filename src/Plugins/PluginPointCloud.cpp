//
// Created by alex on 30.07.24.
//


#include <chrono>

#include "PluginPointCloud.h"
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "Entity.h"
#include "EventsCallbacks.h"
#include "EventsEntity.h"
#include "PointCloudGui.h"
#include "PointCloudIo.h"
#include "Picker.h"
#include "ModuleAABB.h"
#include "ModuleCamera.h"
#include "TransformComponent.h"
#include "GetPrimitives.h"
#include "CommandsAABB.h"
#include "Cuda/BVHCuda.h"
#include "KDTreeCpu.h"
#include "Cuda/Kmeans.h"
#include "Cuda/LocalGaussians.h"
#include "Cuda/Hem.h"
#include "Eigen/Eigenvalues"
#include "ModuleSphereView.h"

namespace Bcg {
    namespace PluginPointCloudInternal {
        static void on_drop_file(const Events::Callback::Drop &event) {
            PluginPointCloud plugin;
            for (int i = 0; i < event.count; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();

                PointCloud pc = PluginPointCloud::load(event.paths[i]);
                if (!pc.interface.is_empty()) {
                    auto end_time = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double> build_duration = end_time - start_time;
                    Log::Info("Build PointCloud in " + std::to_string(build_duration.count()) + " seconds");

                    auto entity_id = Engine::State().create();
                    Engine::State().emplace<PointCloud>(entity_id, pc);
                    Commands::Setup<PointCloud>(entity_id).execute();
                }

            }
        }
    }

    PointCloud PluginPointCloud::load(const std::string &filepath) {
        PointCloud pc;
        if (!Read(filepath, pc)) {
            Log::Error("PointCloud::Unsupported file format: " + filepath);
        }
        return pc;
    }


    PluginPointCloud::PluginPointCloud() : Plugin("PluginPointCloud") {}

    void PluginPointCloud::activate() {
        if (base_activate()) {
            Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&PluginPointCloudInternal::on_drop_file>();
        }
    }

    void PluginPointCloud::begin_frame() {

    }

    void PluginPointCloud::update() {

    }

    void PluginPointCloud::end_frame() {

    }

    void PluginPointCloud::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&PluginPointCloudInternal::on_drop_file>();
        }
    }

    static bool show_pc_gui = false;

    void PluginPointCloud::render_menu() {
        if (ImGui::BeginMenu("Module")) {
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

    namespace Commands {

    }
}