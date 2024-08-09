//
// Created by alex on 30.07.24.
//

#include "PointCloudCommands.h"
#include "EntityCommands.h"
#include "PointCloud.h"
#include "Transform.h"
#include "AABB.h"
#include "Hierarchy.h"
#include "Camera.h"
#include "CameraCommands.h"
#include "SphereViewCommands.h"
#include "KDTreeCpu.h"
#include "GetPrimitives.h"
#include "Eigen/Eigenvalues"
#include "io/io.h"
#include "io/read_xyz.h"
#include "io/read_pts.h"
#include "io/read_csv.h"
#include "Kmeans.h"

namespace Bcg::Commands::Points {
    void LoadPointCloud::execute() const {
        std::string ext = filepath;
        ext = ext.substr(ext.find_last_of('.') + 1);

        PointCloud pc;
        if (ext == "xyz") {
            read_xyz(pc, filepath);
        } else if (ext == "pts") {
            read_pts(pc, filepath);
        } else if (ext == "csv") {
            read_csv(pc, filepath);
        } else {
            Log::Error("Unsupported file format: " + ext);
            return;
        }

        auto entity = entity_id;
        if (!Engine::valid(entity_id)) {
            entity = Engine::State().create();
        }

        Engine::State().emplace_or_replace<PointCloud>(entity, pc);
    }

    void SetupPointCloud::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<PointCloud>(entity_id)) {
            Log::Warn(name + "Entity does not have a PointCloud. Abort Command");
            return;
        }

        auto &pc = Engine::require<PointCloud>(entity_id);
        auto &aabb = Engine::require<AABB>(entity_id);
        auto &transform = Engine::require<Transform>(entity_id);
        auto &hierarchy = Engine::require<Hierarchy>(entity_id);
        Build(aabb, pc.positions());

        Vector<float, 3> center = aabb.center();

        for (auto &point: pc.positions()) {
            point -= center;
        }

        aabb.min -= center;
        aabb.max -= center;

        Commands::View::SetupSphereView(entity_id).execute();

        std::string message = name + ": ";
        message += " #v: " + std::to_string(pc.n_vertices());
        message += " Done.";

        Log::Info(message);
        CenterCameraAtDistance(aabb.center(), aabb.diagonal().maxCoeff()).execute();
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

        auto positions = vertices->get<Vector<float, 3>>("v:point");
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

        auto positions = vertices->get<Vector<float, 3>>("v:point");
        if (!positions) {
            Log::Warn(name + " Entity does not have positions property. Abort Command!");
            return;
        }

        auto result = KMeans(positions.vector(), k);
        auto labels = vertices->get_or_add<unsigned int>("v:kmeans:labels");
        auto distances = vertices->get_or_add<float>("v:kmeans:distances");
        labels.vector() = result.labels;
        distances.vector() = result.distances;
        Engine::State().emplace_or_replace<KMeansResult>(entity_id);
    }
}