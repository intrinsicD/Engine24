//
// Created by alex on 30.07.24.
//


#include "PluginPointCloud.h"
#include "Logger.h"
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "EventsEntity.h"
#include "PointCloudGui.h"
#include <chrono>
#include "PointCloud.h"
#include "PointCloudIo.h"
#include "Picker.h"
#include "PluginAABB.h"
#include "PluginCamera.h"
#include "PluginTransform.h"
#include "PluginHierarchy.h"
#include "GetPrimitives.h"
#include "KDTreeCuda.h"
#include "KDTreeCpu.h"
#include "Kmeans.h"
#include "Eigen/Eigenvalues"
#include "PluginSphereView.h"

namespace Bcg {
    namespace PluginPointCloudInternal {
        static void on_drop_file(const Events::Callback::Drop &event) {
            PluginPointCloud plugin;
            for (int i = 0; i < event.count; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();

                PointCloud pc = PluginPointCloud::load(event.paths[i]);
                if (!pc.is_empty()) {
                    auto end_time = std::chrono::high_resolution_clock::now();

                    std::chrono::duration<double> build_duration = end_time - start_time;
                    Log::Info("Build PointCloud in " + std::to_string(build_duration.count()) + " seconds");

                    auto entity_id = Engine::State().create();
                    Commands::Setup<PointCloud>(entity_id).execute();
                }

            }
        }
    }

    PointCloud PluginPointCloud::load(const std::string &filepath) {
        PointCloud pc;
        if (!Read(filepath, pc)) {
            Log::Error("Unsupported file format: " + filepath);
        }
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

    namespace Commands {
        void Load<PointCloud>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            auto pc = PluginPointCloud::load(filepath);

            if(pc.is_empty()){
                Log::Warn(name + "Failed to load PointCloud from " + filepath);
                return;
            }

            Engine::State().emplace_or_replace<PointCloud>(entity_id, pc);
        }

        void Setup<PointCloud>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            if (!Engine::has<PointCloud>(entity_id)) {
                Log::Warn(name + "Entity does not have a PointCloud. Abort Command");
                return;
            }

            auto &pc = Engine::require<PointCloud>(entity_id);

            Setup<AABB>(entity_id).execute();
            CenterAndScaleByAABB(entity_id, pc.vpoint_.name()).execute();
            auto &aabb = Engine::require<AABB>(entity_id);
            Vector<float, 3> center = aabb.center();

            aabb.min -= center;
            aabb.max -= center;


            auto &transform = Engine::require<Transform>(entity_id);
            auto &hierarchy = Engine::require<Hierarchy>(entity_id);


            Setup<SphereView>(entity_id).execute();

            std::string message = name + ": ";
            message += " #v: " + std::to_string(pc.n_vertices());
            message += " Done.";

            Log::Info(message);
            CenterCameraAtDistance(aabb.center(), aabb.diagonal().maxCoeff()).execute();
        }

        void Cleanup<PointCloud>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn("{}: Entity {} is not valid.", name, entity_id);
                return;
            }

            if (!Engine::has<PointCloud>(entity_id)) {
                Log::Warn("{}: Entity {} does not have Component.", name, entity_id);
                return;
            }

            Engine::Dispatcher().trigger(Events::Entity::PreRemove<PointCloud>{entity_id});
            Engine::State().remove<PointCloud>(entity_id);
            Engine::Dispatcher().trigger(Events::Entity::PostRemove<PointCloud>{entity_id});
            Log::Info("{}: Entity {}", name, entity_id);
        }

        void ComputePointCloudLocalPcasKnn::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + " Entity is not valid. Abort Command!");
                return;
            }

            auto *vertices = GetPrimitives(entity_id).vertices();
            if (!vertices) {
                Log::Warn(name + " Entity does not have vertices. Abort Command!");
                return;
            }

            auto positions = vertices->get<Vector<float, 3>>("v:position");
            if (!positions) {
                Log::Warn(name + " Entity does not have positions property. Abort Command!");
                return;
            }

            if (!Engine::has<KDTreeCpu>(entity_id)) {
                auto &kdtree = Engine::require<KDTreeCpu>(entity_id);
                kdtree.build(positions.vector());
                return;
            }
            auto &kdtree = Engine::require<KDTreeCpu>(entity_id);

            auto evecs0 = vertices->get_or_add<Vector<float, 3>>("v:pca_evecs0", Vector<float, 3>::Unit(0));
            auto evecs1 = vertices->get_or_add<Vector<float, 3>>("v:pca_evecs1", Vector<float, 3>::Unit(1));
            auto evecs2 = vertices->get_or_add<Vector<float, 3>>("v:pca_evecs2", Vector<float, 3>::Unit(2));
            auto evals = vertices->get_or_add<Vector<float, 3>>("v:pca_evals", Vector<float, 3>::Ones());
            for (size_t i = 0; i < vertices->size(); ++i) {
                auto query_point = positions[i];
                auto result = kdtree.knn_query(query_point, num_closest);
                Matrix<float, 3, 3> cov = Matrix<float, 3, 3>::Zero();
                for (auto &knn_idx: result.indices) {
                    Vector<float, 3> diff = positions[knn_idx] - query_point;
                    cov += diff * diff.transpose();
                }

                cov /= (num_closest - 1);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> eigensolver(cov);
                evecs0[i] = eigensolver.eigenvectors().col(0);
                evecs1[i] = eigensolver.eigenvectors().col(1);
                evecs2[i] = eigensolver.eigenvectors().col(2);
                evals[i] = eigensolver.eigenvalues();
            }
        }

        void ComputeKMeans::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + " Entity is not valid. Abort Command!");
                return;
            }

            auto *vertices = GetPrimitives(entity_id).vertices();
            if (!vertices) {
                Log::Warn(name + " Entity does not have vertices. Abort Command!");
                return;
            }

            auto positions = vertices->get<Vector<float, 3>>("v:position");
            if (!positions) {
                Log::Warn(name + " Entity does not have positions property. Abort Command!");
                return;
            }

            auto result = KMeans(positions.vector(), k, iterations);
            auto labels = vertices->get_or_add<unsigned int>("v:kmeans:labels");
            auto distances = vertices->get_or_add<float>("v:kmeans:distances");
            labels.vector() = result.labels;
            distances.vector() = result.distances;
            Engine::State().emplace_or_replace<KMeansResult>(entity_id);
        }

        void ComputeHierarchicalKMeans::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + " Entity is not valid. Abort Command!");
                return;
            }

            auto *vertices = GetPrimitives(entity_id).vertices();
            if (!vertices) {
                Log::Warn(name + " Entity does not have vertices. Abort Command!");
                return;
            }

            auto positions = vertices->get<Vector<float, 3>>("v:position");
            if (!positions) {
                Log::Warn(name + " Entity does not have positions property. Abort Command!");
                return;
            }

            auto result = HierarchicalKMeans(positions.vector(), k, iterations);
            auto labels = vertices->get_or_add<unsigned int>("v:kmeans:labels");
            auto distances = vertices->get_or_add<float>("v:kmeans:distances");
            labels.vector() = result.labels;
            distances.vector() = result.distances;
            Engine::State().emplace_or_replace<KMeansResult>(entity_id);
        }
    }
}