//
// Created by alex on 02.06.25.
//

#include "CommandsMesh.h"
#include "ModuleMesh.h"
#include "SurfaceMeshIo.h"
#include "Engine.h"
#include "Entity.h"
#include "EventsEntity.h"
#include "ModuleAABB.h"
#include "CommandsAABB.h"
#include "ModuleCamera.h"
#include "Hierarchy.h"
#include "ModuleTransform.h"
#include "SurfaceMeshCompute.h"

namespace Bcg::Commands {
    void Load<SurfaceMesh>::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        auto &mesh = Engine::require<SurfaceMesh>(entity_id);

        if (!Read(filepath, mesh)) {
            Log::Warn("Abort {} command", name);
            return;
        }

        if (!mesh.has_face_property("f:indices")) {
            Log::TODO("Implement: Mesh does not have faces, its a Point Cloud. Forward to Point Cloud stuff...");
        }
    }

    void Setup<SurfaceMesh>::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<SurfaceMesh>(entity_id)) {
            Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
        }

        auto &mesh = Engine::require<SurfaceMesh>(entity_id);


        ModuleAABB::setup(entity_id);
        CenterAndScaleByAABB(entity_id, mesh.vpoint_.name()).execute();

        auto h_aabb = Engine::State().get<AABBHandle>(entity_id);
        auto &transform = *ModuleTransform::setup(entity_id);
        auto &hierarchy = Engine::require<Hierarchy>(entity_id);

        std::string message = name + ": ";
        message += " #v: " + std::to_string(mesh.n_vertices());
        message += " #e: " + std::to_string(mesh.n_edges());
        message += " #h: " + std::to_string(mesh.n_halfedges());
        message += " #f: " + std::to_string(mesh.n_faces());
        message += " Done.";

        Log::Info(message);
        float d = 1.5 * glm::compMax(h_aabb->diagonal());
        ModuleCamera::center_camera_at_distance(h_aabb->center(), d);
        ComputeSurfaceMeshVertexNormals(entity_id);
    }

    void Cleanup<SurfaceMesh>::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Warn(name + "Entity is not valid. Abort Command");
            return;
        }

        if (!Engine::has<SurfaceMesh>(entity_id)) {
            Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
            return;
        }

        Engine::Dispatcher().trigger(Events::Entity::PreRemove<SurfaceMesh>{entity_id});
        Engine::State().remove<SurfaceMesh>(entity_id);
        Engine::Dispatcher().trigger(Events::Entity::PostRemove<SurfaceMesh>{entity_id});
        Log::Info("{} for entity {}", name, entity_id);
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