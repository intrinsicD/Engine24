//
// Created by alex on 12.08.25.
//

#include "CommandsPointCloud.h"
#include "Engine.h"
#include "PointCloudIo.h"
#include "ModuleAABB.h"
#include "CommandsAABB.h"
#include "TransformComponent.h"
#include "ModuleSphereView.h"
#include "ModuleCamera.h"
#include "EventsEntity.h"
#include "GetPrimitives.h"
#include "ModulePointCloud.h"
#include "KDTreeCpu.h"
#include "Entity.h"
#include "Cuda/LocalGaussians.h"
#include "Cuda/Kmeans.h"
#include "Cuda/Hem.h"
#include "Eigen/Eigenvalues"

namespace Bcg::Commands{
    void Load<PointCloud>::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        auto pc = Engine::require<PointCloud>(entity_id);

        if (!Read(filepath, pc)) {
            Log::Warn(name + "Failed to load PointCloud from " + filepath);
            Engine::State().remove<PointCloud>(entity_id);
            return;
        }
    }

    void Setup<PointCloud>::execute() const {
        /*if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<PointCloud>(entity_id)) {
            Log::Warn(name + "Entity does not have a PointCloud. Abort Command");
            return;
        }

        auto &pc = Engine::require<PointCloud>(entity_id);

        ModuleAABB::setup(entity_id);
        CenterAndScaleByAABB(entity_id, pc.vpoint_.name()).execute();
        auto h_aabb = ModuleAABB::get(entity_id);
        Vector<float, 3> c = h_aabb->center();

        h_aabb->min -= c;
        h_aabb->max -= c;

        auto &transform = Engine::require<TransformComponent>(entity_id);

        ModuleSphereView::setup(entity_id);

        std::string message = name + ": ";
        message += " #v: " + std::to_string(pc.n_vertices());
        message += " Done.";

        Log::Info(message);
        ModuleCamera::center_camera_at_distance(c, glm::compMax(h_aabb->diagonal()));*/

        ModulePointCloud::setup(entity_id);
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
            for (long i = 0; i < result.indices.size(); ++i) {
                Vector<float, 3> diff = positions[result.indices[i]] - query_point;
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