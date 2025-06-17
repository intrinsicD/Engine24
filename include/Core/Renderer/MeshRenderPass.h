//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_MESHRENDERPASS_H
#define ENGINE24_MESHRENDERPASS_H

#include "IRenderPass.h"
#include "AssetManager.h"
#include "Program.h"

namespace Bcg{
    class MeshRenderPass : public IRenderPass {
    public:
        MeshRenderPass(AssetManager& asset_manager);

        void execute(entt::registry& registry, const Camera& camera) override;
    private:
        AssetManager& m_asset_manager;
        Program m_static_mesh_shader;
        // ... other resources needed specifically for mesh rendering
    };
}
#endif //ENGINE24_MESHRENDERPASS_H
