//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_RENDERABLEMESHCOMPONENT_H
#define ENGINE24_RENDERABLEMESHCOMPONENT_H

#include <vector>
#include "AssetHandle.h"

namespace Bcg {

    struct RenderableMeshComponent {
        // Handle to the Mesh asset, which may contain one or more sub-meshes.
        AssetHandle mesh_handle;

        // A vector of material handles. The order of materials in this vector
        // should correspond to the order of sub-meshes within the Mesh asset.
        // This allows different parts of a single mesh to be rendered with
        // different materials (e.g., a character model with different materials
        // for skin, armor, and cloth).
        std::vector<AssetHandle> material_handles;

        // A simple flag to control the visibility of the entity.
        // The rendering system can quickly check this flag to skip drawing
        // without needing to remove any components.
        bool visible = true;
    };
}

#endif //ENGINE24_RENDERABLEMESHCOMPONENT_H
