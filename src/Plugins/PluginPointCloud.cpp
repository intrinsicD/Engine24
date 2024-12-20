//
// Created by alex on 30.07.24.
//


#include <chrono>

#include "PluginPointCloud.h"
#include "Logger.h"
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "Entity.h"
#include "EventsCallbacks.h"
#include "EventsEntity.h"
#include "PointCloudGui.h"
#include "PointCloud.h"
#include "PointCloudIo.h"
#include "Picker.h"
#include "BoundingVolumes.h"
#include "PluginAABB.h"
#include "PluginCamera.h"
#include "PluginTransform.h"
#include "PluginHierarchy.h"
#include "GetPrimitives.h"
#include "Cuda/KDTreeCuda.h"
#include "KDTreeCpu.h"
#include "Cuda/Kmeans.h"
#include "Cuda/LocalGaussians.h"
#include "Cuda/Hem.h"
#include "Eigen/Eigenvalues"
#include "PluginViewSphere.h"

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
                    Engine::State().emplace<PointCloud>(entity_id, pc);
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

            if (pc.is_empty()) {
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
            auto &bv = Engine::State().get<BoundingVolumes>(entity_id);
            auto &aabb = *bv.h_aabb;
            Vector<float, 3> c = aabb.center();

            aabb.min -= c;
            aabb.max -= c;


            auto &transform = Engine::require<Transform>(entity_id);
            auto &hierarchy = Engine::require<Hierarchy>(entity_id);


            Setup<SphereView>(entity_id).execute();

            std::string message = name + ": ";
            message += " #v: " + std::to_string(pc.n_vertices());
            message += " Done.";

            Log::Info(message);
            CenterCameraAtDistance(c, glm::compMax(aabb.diagonal())).execute();
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

            auto evecs0 = vertices->get_or_add<Vector<float, 3>>("v:pca_evecs0", Vector<float, 3>(1.0f, 0.0f, 0.0f));
            auto evecs1 = vertices->get_or_add<Vector<float, 3>>("v:pca_evecs1", Vector<float, 3>(0.0f, 1.0f, 0.0f));
            auto evecs2 = vertices->get_or_add<Vector<float, 3>>("v:pca_evecs2", Vector<float, 3>(0.0f, 0.0f, 1.0f));
            auto evals = vertices->get_or_add<Vector<float, 3>>("v:pca_evals", Vector<float, 3>(1.0f));
            for (size_t i = 0; i < vertices->size(); ++i) {
                auto query_point = positions[i];
                auto result = kdtree.knn_query(query_point, num_closest);
                Matrix<float, 3, 3> cov = Matrix<float, 3, 3>(0.0f);
                for (long i = 0; i < result.indices.row(0).size(); ++i) {
                    Vector<float, 3> diff = positions[result.indices(0, i)] - query_point;
                    cov += glm::outerProduct(diff, diff);
                }

                cov /= num_closest;

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> eigensolver(MapConst(cov));
                Map(evecs0[i]) = eigensolver.eigenvectors().col(0);
                Map(evecs1[i]) = eigensolver.eigenvectors().col(1);
                Map(evecs2[i]) = eigensolver.eigenvectors().col(2);
                Map(evals[i]) = eigensolver.eigenvalues();
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

            auto result = cuda::KMeans(positions.vector(), k, iterations);
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

            auto result = cuda::HierarchicalKMeans(positions.vector(), k, iterations);
            auto labels = vertices->get_or_add<unsigned int>("v:kmeans:labels");
            auto distances = vertices->get_or_add<float>("v:kmeans:distances");
            labels.vector() = result.labels;
            distances.vector() = result.distances;
            Engine::State().emplace_or_replace<KMeansResult>(entity_id);
        }

        void ComputeLocalGaussians::execute() const {
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
            auto result = cuda::LocalGaussians(positions.vector(), num_closest);
            auto means = vertices->get_or_add<Vector<float, 3>>("v:local_gaussians:means");
            auto covs = vertices->get_or_add<Matrix<float, 3, 3>>("v:local_gaussians:covs");
            auto evecs0 = vertices->get_or_add<Vector<float, 3>>("v:local_gaussians:evecs0");
            auto evecs1 = vertices->get_or_add<Vector<float, 3>>("v:local_gaussians:evecs1");
            auto evecs2 = vertices->get_or_add<Vector<float, 3>>("v:local_gaussians:evecs2");
            covs.vector() = result.covs;
            means.vector() = result.means;

            for (size_t i = 0; i < vertices->size(); ++i) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> eigensolver(MapConst(result.covs[i]));
                Map(evecs0[i]) = eigensolver.eigenvectors().col(0);
                Map(evecs1[i]) = eigensolver.eigenvectors().col(1);
                Map(evecs2[i]) = eigensolver.eigenvectors().col(2);
            }
        }

        void ComputeHem::execute() const {
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
            auto result = cuda::Hem(positions.vector(), levels, num_closest);
            entt::entity entity_id = Engine::State().create();
            auto &hem = Engine::State().emplace<PointCloud>(entity_id);
            for(const auto &mean : result.means){
                hem.add_vertex(mean);
            }

            auto means = hem.vertex_property<Vector<float, 3>>("v:hem:means");
            auto covs = hem.vertex_property<Matrix<float, 3, 3>>("v:hem:covs");
            auto nvars = hem.vertex_property<Vector<float, 3>>("v:hem:nvars");
            auto weights = hem.vertex_property<float>("v:hem:weigths");
            auto evecs0 = hem.vertex_property<Vector<float, 3>>("v:hem:evecs0");
            auto evecs1 = hem.vertex_property<Vector<float, 3>>("v:hem:evecs1");
            auto evecs2 = hem.vertex_property<Vector<float, 3>>("v:hem:evecs2");

            covs.vector() = result.covs;
            means.vector() = result.means;
            nvars.vector() = result.nvars;
            weights.vector() = result.weights;

            for (size_t i = 0; i < hem.n_vertices(); ++i) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3>> eigensolver(MapConst(covs.vector()[i]));
                Map(evecs0.vector()[i]) = eigensolver.eigenvectors().col(0);
                Map(evecs1.vector()[i]) = eigensolver.eigenvectors().col(1);
                Map(evecs2.vector()[i]) = eigensolver.eigenvectors().col(2);
            }

            Setup<PointCloud>(entity_id).execute();
        }
    }
}