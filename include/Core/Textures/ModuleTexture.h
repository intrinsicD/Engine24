//
// Created by alex on 13.06.25.
//

#ifndef ENGINE24_MODULETEXTURE_H
#define ENGINE24_MODULETEXTURE_H

#include "Module.h"

namespace Bcg {
    using TextureHandle = uint32_t;
    const TextureHandle InvalidTextureHandle = 0;

    struct TextureCreateDescriptor {
        std::string name;
        unsigned int width;
        unsigned int height;
        unsigned int channels;

        const void *data;
    };

    struct TextureInfo{
        std::string name;
        unsigned int width;
        unsigned int height;
        unsigned int channels;
        unsigned int id; // OpenGL texture ID
        unsigned int target; // OpenGL texture target (e.g., GL_TEXTURE_2D)
    };

    class ModuleTexture : public Module {
    public:
        ModuleTexture() : Module("ModuleTexture") {}

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        TextureHandle get_handle(const std::string &name) {
            auto it = handle_lookup.find(name);
            if (it != handle_lookup.end()) {
                return it->second;
            }
            return InvalidTextureHandle;
        }

        TextureHandle load(const std::string &file_path, const std::string &name = "");

        TextureHandle create(const TextureCreateDescriptor &descriptor) {
            TextureHandle handle = get_handle(descriptor.name);
            if(handle != InvalidTextureHandle) {
                return handle; // Texture already exists
            }
            // Create a new texture handle
            handle = m_NextHandle++;
            TextureInfo texture_info;
            texture_info.name = descriptor.name;
            texture_info.width = descriptor.width;
            texture_info.height = descriptor.height;
            texture_info.channels = descriptor.channels;
            texture_info.id = 0; // OpenGL texture ID will be set later
            texture_info.target = 0; //will be set later
            textures_registry[handle] = texture_info;
            handle_lookup[descriptor.name] = handle; // Add to lookup by name
            return handle;
        }

        TextureInfo *get(TextureHandle handle) {
            auto it = textures_registry.find(handle);
            if (it != textures_registry.end()) {
                return &it->second;
            }
            return nullptr;
        }

        TextureInfo *get(const std::string &name) {
            auto it = handle_lookup.find(name);
            if (it != handle_lookup.end()) {
                return get(it->second);
            }
            return nullptr;
        }

        void free_loading_cache() { // Clear the cache of textures that are loaded if memory is low
            loading_cache.clear();
        }

    protected:
        std::unordered_map<std::string, TextureHandle> handle_lookup; // lookup by name of texture handles
        std::unordered_map<TextureHandle, TextureInfo> textures_registry; // lookup by handle of texture instances
        std::unordered_map<std::string, TextureCreateDescriptor> loading_cache; //caches texture data for loaded files to avoid reloading
        TextureHandle m_NextHandle = 1;
    };
}

#endif //ENGINE24_MODULETEXTURE_H
