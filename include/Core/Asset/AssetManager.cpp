//
// Created by alex on 17.06.25.
//

#include "AssetManager.h"
#include <thread> // For std::this_thread

namespace Bcg {
    AssetManager::AssetManager() {}
    AssetManager::~AssetManager() {}

    void AssetManager::register_loader(std::shared_ptr<IAssetLoader> loader) {
        std::lock_guard<std::mutex> lock(m_asset_mutex);
        for (const auto& ext : loader->get_supported_extensions()) {
            m_loaders[ext] = loader;
        }
    }

    AssetHandle AssetManager::load_asset(const std::string &path) {
        AssetID id = GenerateAssetID(path);

        { // Scoped lock
            std::lock_guard<std::mutex> lock(m_asset_mutex);
            // If asset is already loaded or is currently being loaded, return handle.
            if (m_asset_registry.count(id) || m_pending_assets.count(id)) {
                return AssetHandle(id);
            }

            std::string extension = get_file_extension(path);
            if (!m_loaders.count(extension)) {
                // Log error: No loader for this file type
                return AssetHandle{}; // Return invalid handle
            }

            // Store path/ID mapping
            m_path_to_id[path] = id;
            m_id_to_path[id] = path;

            auto loader = m_loaders.at(extension);

            // Launch the loading task asynchronously.
            // std::async is simple. A dedicated thread pool/job system is better for a large engine.
            m_pending_assets[id] = std::async(std::launch::async, [loader, path, id]() {
                auto asset = loader->load_asset(path);
                if (asset) {
                    asset->handle = AssetHandle(id);
                }
                return asset;
            });
        }

        return AssetHandle(id);
    }

    void AssetManager::process_pending_asset(Bcg::AssetID id) {
        std::shared_ptr<IAsset> asset = nullptr;

        // Check if the asset is in the pending map.
        m_asset_mutex.lock();
        if (m_pending_assets.count(id)) {
            // .get() will block until the future is ready.
            asset = m_pending_assets.at(id).get();
            m_pending_assets.erase(id); // Remove from pending list
            m_asset_mutex.unlock();
        } else {
            m_asset_mutex.unlock();
            return; // Not a pending asset
        }

        // If loading was successful, move the asset to the main registry.
        if (asset) {
            std::lock_guard<std::mutex> lock(m_asset_mutex);
            m_asset_registry[id] = asset;
        }
    }

    bool AssetManager::is_asset_loaded(Bcg::AssetHandle handle) {
        if (!handle.is_valid()) return false;

        AssetID id = handle.get_id();
        std::lock_guard<std::mutex> lock(m_asset_mutex);

        if (m_asset_registry.count(id)) {
            return true;
        }

        if (m_pending_assets.count(id)) {
            // Check if the future is ready without blocking.
            auto& future = m_pending_assets.at(id);
            return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }
        return false;
    }

    void AssetManager::reload_asset(Bcg::AssetHandle handle) {
        if (!handle.is_valid()) return;
        AssetID id = handle.get_id();

        std::string path;
        {
            std::lock_guard<std::mutex> lock(m_asset_mutex);
            if (!m_id_to_path.count(id)) return; // Don't know this asset
            path = m_id_to_path.at(id);

            // Remove from registry to force a reload. The shared_ptr ref count will handle memory.
            m_asset_registry.erase(id);
        }

        // Request a load just like a new asset.
        load_asset(path);
    }

    std::string AssetManager::get_file_extension(const std::string &path) {
        size_t pos = path.rfind('.');
        return (pos != std::string::npos) ? path.substr(pos) : "";
    }
}