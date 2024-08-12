//
// Created by alex on 18.07.24.
//

#include "MeshCommands.h"
#include "EntityCommands.h"
#include "SurfaceMesh.h"
#include "AABB.h"
#include "Hierarchy.h"
#include "CameraCommands.h"
#include "MeshCompute.h"
#include "PluginTransform.h"

namespace Bcg::Commands {
    void Setup<SurfaceMesh>::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<SurfaceMesh>(entity_id)) {
            Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
        }

        auto &mesh = Engine::require<SurfaceMesh>(entity_id);
        auto &aabb = Engine::require<AABB>(entity_id);
        auto &transform = *PluginTransform::setup(entity_id);
        auto &hierarchy = Engine::require<Hierarchy>(entity_id);

        Build(aabb, mesh.positions());

        Vector<float, 3> center = aabb.center();
        float scale = (aabb.max - aabb.min).maxCoeff();

        for (auto &point: mesh.positions()) {
            point -= center;
            point /= scale;
        }

        aabb.min = (aabb.min - center) / scale;
        aabb.max = (aabb.max - center) / scale;

        std::string message = name + ": ";
        message += " #v: " + std::to_string(mesh.n_vertices());
        message += " #e: " + std::to_string(mesh.n_edges());
        message += " #h: " + std::to_string(mesh.n_halfedges());
        message += " #f: " + std::to_string(mesh.n_faces());
        message += " Done.";

        Log::Info(message);
        float d = aabb.diagonal().maxCoeff();
        CenterCameraAtDistance(aabb.center(), d).execute();
        ComputeSurfaceMeshVertexNormals(entity_id);
    }


    void ComputeFaceNormals::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<SurfaceMesh>(entity_id)) {
            Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
            return;
        }

        auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);

/*        auto v_normals = ComputeFaceNormals(entity_id, mesh);*/
    }
}