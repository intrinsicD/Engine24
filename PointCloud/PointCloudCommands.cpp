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

namespace Bcg::Commands::Points {
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
}