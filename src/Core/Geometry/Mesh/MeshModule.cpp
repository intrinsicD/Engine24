//
// Created by alex on 26.11.24.
//

#include "MeshModule.h"
#include "ModuleCamera.h"
#include "ModuleTransform.h"

#include "ModuleAABB.h"
#include "MeshGuiModule.h"
#include "Pool.h"
#include "Engine.h"
#include "MeshResources.h"
#include "SurfaceMeshIo.h"

namespace Bcg {
    template<>
    struct StringTraits<SurfaceMesh> {
        static std::string ToString(const SurfaceMesh &t) {
            return "Mesh to string not jet implemented";
        }
    };

    MeshModule::MeshModule() : Module("MeshModule") {
    }

    void MeshModule::activate() {
        if (base_activate()) {
            if (!Engine::Context().find<MeshPool>()) {
                Engine::Context().emplace<MeshPool>();
            }
        }
        MeshResources::activate();
    }

    void MeshModule::deactivate() {
        if (base_deactivate()) {
            if (Engine::Context().find<MeshPool>()) {
                Engine::Context().erase<MeshPool>();
            }
        }
    }

    MeshHandle MeshModule::make_handle(const SurfaceMesh &object) {
        auto &pool = Engine::Context().get<MeshPool>();
        return pool.create(object);
    }

    struct AABBGetter{
        AABB operator()(const SurfaceMesh &mesh) const {
            return AABB::Build(mesh.positions().begin(), mesh.positions().end());
        }
    };

    static std::string s_name = "MeshModule";

    MeshHandle MeshModule::create(entt::entity entity_id, const SurfaceMesh &object) {
        auto handle = make_handle(object);
        add(entity_id, handle);
        //TODO setup everything else what should happen when a mesh is created... (seutp rendering, build kdtree, aabb, transform etc...)

        auto h_aabb = ModuleAABB::create(entity_id, AABBGetter()(handle));
        auto h_transform = ModuleTransform::create(entity_id, Transform::Identity());

        ModuleAABB::center_and_scale_by_aabb(entity_id, handle->vpoint_.name());
        ModuleCamera::center_camera_at_distance(h_aabb->center(), 1.5f * glm::compMax(h_aabb->diagonal()));

        std::string message = s_name + ": ";
        message += " #v: " + std::to_string(handle->n_vertices());
        message += " #e: " + std::to_string(handle->n_edges());
        message += " #h: " + std::to_string(handle->n_halfedges());
        message += " #f: " + std::to_string(handle->n_faces());
        message += " Done.";

        Log::Info(message);
        return handle;
    }

    MeshHandle MeshModule::add(entt::entity entity_id, const MeshHandle h_mesh) {
        return Engine::State().get_or_emplace<MeshHandle>(entity_id, h_mesh);
    }

    void MeshModule::remove(entt::entity entity_id) {
        Engine::State().remove<MeshHandle>(entity_id);
    }

    bool MeshModule::has(entt::entity entity_id) {
        return Engine::State().all_of<MeshHandle>(entity_id);
    }

    MeshHandle MeshModule::get(entt::entity entity_id) {
        return Engine::State().get<MeshHandle>(entity_id);
    }

    SurfaceMesh MeshModule::load_mesh(const std::string &filepath) {
        return MeshResources::load(filepath);
    }

    bool MeshModule::save_mesh(const std::string &filepath, const SurfaceMesh &mesh) {
        if (!Write(filepath, mesh)) {
            std::string ext = filepath;
            ext = ext.substr(ext.find_last_of('.') + 1);
            Log::Error("Unsupported file format: " + ext);
            return false;
        }
        return true;
    }
}
