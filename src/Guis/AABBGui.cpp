//
// Created by alex on 28.07.24.
//

#include "AABBGui.h"
#include "AABBUtils.h"
#include "Engine.h"
#include "imgui.h"
#include "BoundingVolumes.h"

namespace Bcg::Gui {
    void Show(const PoolHandle<AABB<float, 3>> &h_aabb) {
        if (h_aabb.is_valid()) {
            Show(*h_aabb);
        }
    }

    void Show(const AABB<float, 3> &aabb) {
        ImGui::Text("%s", StringTraits<AABB<float, 3>>::ToString(aabb).c_str());
    }

    void Show(entt::entity entity_id) {
        if (Engine::State().all_of<BoundingVolumes>(entity_id)) {
            auto &bv = Engine::State().get<BoundingVolumes>(entity_id);
            Show(bv.h_aabb);
        }
    }
}
