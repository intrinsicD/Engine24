//
// Created by alex on 26.11.24.
//

#include "MeshModule.h"
#include "MeshGuiModule.h"
#include "Logger.h"
#include "Pool.h"
#include "Engine.h"
#include "MeshComponent.h"
#include "MeshResources.h"
#include "SurfaceMeshIo.h"
#include "StringTraits.h"

namespace Bcg {
    template<>
    struct StringTraits<SurfaceMesh> {
        static std::string ToString(const SurfaceMesh &t) {
            return "Mesh to string not jet implemented";
        }
    };

    MeshModule::MeshModule() : ComponentModule("MeshModule") {
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

    PoolHandle<SurfaceMesh> MeshModule::make_handle(const SurfaceMesh &mesh) {
        auto &pool = Engine::Context().get<MeshPool>();
        return pool.create(mesh);
    }

    PoolHandle<SurfaceMesh> MeshModule::create(entt::entity entity_id, const SurfaceMesh &mesh) {
        auto &pool = Engine::Context().get<MeshPool>();
        auto h_mesh = pool.create(mesh);
        add(entity_id, h_mesh);
        //TODO setup everything else what should happen when a mesh is created... (seutp rendering, build kdtree, aabb, transform etc...)
        return h_mesh;
    }

    PoolHandle<SurfaceMesh> MeshModule::add(entt::entity entity_id, const PoolHandle<SurfaceMesh> h_mesh) {
        auto &pool = Engine::Context().get<MeshPool>();
        auto &meshes = Engine::State().get_or_emplace<MeshComponent>(entity_id);
        meshes.meshes.push_back(h_mesh);
        meshes.current_mesh = h_mesh;
        return h_mesh;
    }

    void MeshModule::remove(entt::entity entity_id) {
        Engine::State().remove<MeshComponent>(entity_id);
    }

    bool MeshModule::has(entt::entity entity_id) {
        return Engine::State().all_of<MeshComponent>(entity_id);
    }

    PoolHandle<SurfaceMesh> MeshModule::get(entt::entity entity_id) {
        auto &meshes = Engine::State().get<MeshComponent>(entity_id);
        return meshes.current_mesh;
    }

    PoolHandle<SurfaceMesh> MeshModule::load_mesh(const std::string &filepath) {
        auto ret = MeshResources::load(filepath);

        auto &pool = Engine::Context().get<MeshPool>();
        return pool.create(ret);
    }
}
