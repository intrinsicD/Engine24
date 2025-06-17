//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_ASSETMANAGER_H
#define ENGINE24_ASSETMANAGER_H

#include "IAssetLoader.h"
#include <future>

namespace Bcg {
    inline AssetID GenerateAssetID(const std::string &path) {
        return std::hash<std::string>{}(path);
    }

    class AssetManager {
    public:
        AssetManager();

        ~AssetManager();

        // Registers a loader for a specific set of file extensions.
        void register_loader(std::shared_ptr<IAssetLoader> loader);

        // Asynchronously loads an asset from a file path. Returns a handle immediately.
        AssetHandle load_asset(const std::string &path);

        // Gets the asset pointer from a handle.
        // This will block if the asset is still loading.
        template<typename T>
        std::shared_ptr<T> get_asset(AssetHandle handle);

        // Non-blocking check to see if an asset is loaded and ready.
        bool is_asset_loaded(AssetHandle handle);

        // For hot-reloading: tells the manager to reload an asset from its path.
        void reload_asset(AssetHandle handle);

    private:
        // Helper to get the file extension from a path.
        std::string get_file_extension(const std::string &path);

        // Checks if an asset is currently loading and, if so, waits for it to finish.
        // Moves the loaded asset from the pending map to the main registry.
        void process_pending_asset(AssetID id);

        std::mutex m_asset_mutex;

        // Maps file extensions (e.g., ".png") to the correct loader.
        std::unordered_map<std::string, std::shared_ptr<IAssetLoader>> m_loaders;

        // Maps a path string to its generated AssetID.
        std::unordered_map<std::string, AssetID> m_path_to_id;
        // Maps an AssetID back to its original path (needed for reloading).
        std::unordered_map<AssetID, std::string> m_id_to_path;

        // The main cache of loaded assets.
        std::unordered_map<AssetID, std::shared_ptr<IAsset>> m_asset_registry;

        // A map of assets that are currently being loaded in the background.
        std::unordered_map<AssetID, std::future<std::shared_ptr<IAsset>>> m_pending_assets;
    };

    // Template implementation must be in the header.
    template<typename T>
    std::shared_ptr<T> AssetManager::get_asset(AssetHandle handle) {
        if (!handle.is_valid()) {
            return nullptr;
        }

        AssetID id = handle.get_id();

        // If the asset is pending, this call will block until it's loaded.
        process_pending_asset(id);

        std::lock_guard<std::mutex> lock(m_asset_mutex);
        if (m_asset_registry.count(id)) {
            // Use dynamic_pointer_cast for safe downcasting from IAsset to the derived type.
            return std::dynamic_pointer_cast<T>(m_asset_registry.at(id));
        }

        return nullptr;
    }
}

#endif //ENGINE24_ASSETMANAGER_H
