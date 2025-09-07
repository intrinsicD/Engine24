//
// Created by alex on 6/17/25.
//

#ifndef VECTORFIELDRENDERPASS_H
#define VECTORFIELDRENDERPASS_H

#include "IRenderPass.h"
#include "AssetManager.h"
#include "Program.h"

namespace Bcg{
    class VectorfieldRenderPass : public IRenderPass {
    public:
        VectorfieldRenderPass(AssetManager& asset_manager);

        void execute(entt::registry& registry, const Camera& camera) override;
    private:
        AssetManager& m_asset_manager;
        Program m_static_mesh_shader;
        // ... other resources needed specifically for mesh rendering
    };
}
#endif //VECTORFIELDRENDERPASS_H
