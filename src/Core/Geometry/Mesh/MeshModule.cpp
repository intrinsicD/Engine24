//
// Created by alex on 26.11.24.
//

#include "MeshModule.h"
#include "MeshGuiModule.h"
#include "Logger.h"
#include "Pool.h"
#include "Engine.h"
#include "MeshComponent.h"
#include "SurfaceMeshIo.h"
#include "Cache.h"
#include "StringTraits.h"

namespace Bcg {

    template<>
    struct StringTraits<SurfaceMesh> {
        static std::string ToString(const SurfaceMesh &t) {
            return "Mesh to string not jet implemented";
        }
    };

    void MeshModule::activate() {
        if (!Engine::Context().find<Pool<SurfaceMesh> >()) {
            Engine::Context().emplace<Pool<SurfaceMesh> >();
        }
        if(!Engine::Context().find<Cache<SurfaceMesh> >()) {
            Engine::Context().emplace<Cache<SurfaceMesh> >();
        }
        Log::Info(name + " activated");
    }

    void MeshModule::deactivate() {
        if (Engine::Context().find<Pool<SurfaceMesh> >()) {
            Engine::Context().erase<Pool<SurfaceMesh> >();
        }
        if(Engine::Context().find<Cache<SurfaceMesh> >()) {
            Engine::Context().erase<Cache<SurfaceMesh> >();
        }
        Log::Info(name + " deactivated");
    }

    PoolHandle<SurfaceMesh> MeshModule::make_handle(const SurfaceMesh &mesh) {
        auto &pool = Engine::Context().get<Pool<SurfaceMesh> >();
        return pool.create(mesh);
    }

    PoolHandle<SurfaceMesh> MeshModule::create(entt::entity entity_id, const SurfaceMesh &mesh) {
        auto &pool = Engine::Context().get<Pool<SurfaceMesh> >();
        auto h_mesh = pool.create(mesh);
        add(entity_id, h_mesh);
        return h_mesh;
    }

    PoolHandle<SurfaceMesh> MeshModule::add(entt::entity entity_id, const PoolHandle<SurfaceMesh> h_mesh) {
        auto &pool = Engine::Context().get<Pool<SurfaceMesh> >();
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
        auto &cache = Engine::Context().get<Cache<SurfaceMesh> >();
        SurfaceMesh mesh;
        if(cache.contains(filepath)) {
            mesh =  cache[filepath];
        }else{
            Read(filepath, mesh);
            cache[filepath] = mesh;
        }
        auto &pool = Engine::Context().get<Pool<SurfaceMesh> >();
        return pool.create(mesh);
    }

    VertexProperty<Vector<float, 3>> MeshModule::compute_vertex_normals(SurfaceMesh &mesh) {

    }

    FaceProperty<Vector<float, 3>> MeshModule::compute_face_normals(SurfaceMesh &mesh) {

    }

    FaceProperty<Vector<float, 3>> MeshModule::compute_face_centers(SurfaceMesh &mesh) {

    }

    EdgeProperty<float> MeshModule::compute_edge_lengths(SurfaceMesh &mesh) {

    }

    size_t MeshModule::get_num_connected_components(SurfaceMesh &mesh) {

    }

    std::vector<SurfaceMesh> MeshModule::split_connected_components(SurfaceMesh &mesh) {

    }

    SurfaceMesh MeshModule::merge_meshes(const std::vector<SurfaceMesh> &meshes) {

    }
}

