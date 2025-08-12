//
// Created by alex on 5/28/25.
//

#include "ResourcesPointCloud.h"
#include "PointCloudIo.h"
#include "Engine.h"
#include "entt/resource/resource.hpp"
#include "entt/resource/cache.hpp"
#include "entt/resource/loader.hpp"

#include <chrono>

namespace Bcg {
    struct PointCloudLoader : entt::resource_loader<PointCloud> {
        using result_type = std::shared_ptr<PointCloud>;

        result_type operator()(const std::string &filepath) const {
            result_type result = std::make_shared<PointCloud>();
            Read(filepath, *result);
            return std::make_shared<PointCloud>(*result);
        }
    };

    using PointCloudAssetCache = entt::resource_cache<PointCloud, PointCloudLoader>;

    void ResourcesPointCloud::activate() {
        Engine::Context().emplace<entt::resource_cache<PointCloud, PointCloudLoader>>();
        Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&ResourcesPointCloud::on_drop_file>();
    }

    PointCloud ResourcesPointCloud::load(const std::string &filepath) {
        auto &cache = Engine::Context().get<PointCloudAssetCache>();
        auto ret = cache.load(entt::hashed_string(filepath.c_str()), filepath);
        const bool loaded = ret.second;
        if (loaded) {
            Log::Info(("Loading point cloud from file: " + filepath).c_str());
        } else {
            Log::Info(("Loaded point cloud from cache: " + filepath).c_str());
        }
        return ret.first->second;
    }

    void ResourcesPointCloud::on_drop_file(const Events::Callback::Drop &event) {
        for (int i = 0; i < event.count; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            PointCloud spc = load(event.paths[i]);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> build_duration = end_time - start_time;
            Log::Info("Loading pc took: " + std::to_string(build_duration.count()) + " seconds");
        }
    }
}