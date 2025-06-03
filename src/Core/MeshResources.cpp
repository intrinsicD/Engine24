//
// Created by alex on 5/28/25.
//

#include "ResourcesMesh.h"
#include "SurfaceMeshIo.h"
#include "Logger.h"
#include "Engine.h"
#include "entt/resource/resource.hpp"
#include "entt/resource/cache.hpp"
#include "entt/resource/loader.hpp"
#include "entt/signal/dispatcher.hpp"

namespace Bcg {
    struct SurfaceMeshLoader : entt::resource_loader<SurfaceMesh> {
        using result_type = std::shared_ptr<SurfaceMesh>;

        result_type operator()(const std::string &filepath) const {
            result_type result = std::make_shared<SurfaceMesh>();
            Read(filepath, *result);
            return std::make_shared<SurfaceMesh>(*result);
        }
    };

    using MeshAssetCache = entt::resource_cache<SurfaceMesh, SurfaceMeshLoader>;

    void ResourcesMesh::activate() {
        Engine::Context().emplace<entt::resource_cache<SurfaceMesh, SurfaceMeshLoader>>();
        Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&ResourcesMesh::on_drop_file>();
    }

    SurfaceMesh ResourcesMesh::load(const std::string &filepath) {
        auto &cache = Engine::Context().get<MeshAssetCache>();
        auto ret = cache.load(entt::hashed_string(filepath.c_str()), filepath);
        const bool loaded = ret.second;
        if (loaded) {
            Log::Info(("Loading mesh from file: " + filepath).c_str());
        } else {
            Log::Info(("Loaded mesh from cache: " + filepath).c_str());
        }
        return ret.first->second;
    }

    void ResourcesMesh::on_drop_file(const Events::Callback::Drop &event) {
        for (int i = 0; i < event.count; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            SurfaceMesh smesh = load(event.paths[i]);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> build_duration = end_time - start_time;
            Log::Info("Loading mesh took: " + std::to_string(build_duration.count()) + " seconds");
        }
    }


}