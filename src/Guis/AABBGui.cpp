//
// Created by alex on 28.07.24.
//

#include "AABBGui.h"
#include "Engine.h"
#include "imgui.h"
#include "BoundingVolumes.h"

namespace Bcg::Gui {
    void Show(const PoolHandle<AABB> &h_aabb) {
        if (h_aabb.is_valid()) {
            Show(*h_aabb);
        }
    }

    void Show(const AABB &aabb) {
        ImGui::Text("Min: (%f, %f, %f)", aabb.min.x, aabb.min.y, aabb.min.z);
        ImGui::Text("Max: (%f, %f, %f)", aabb.max.x, aabb.max.y, aabb.max.z);
    }

    void Show(entt::entity entity_id) {
        if (Engine::State().all_of<BoundingVolumes>(entity_id)) {
            auto &bv = Engine::State().get<BoundingVolumes>(entity_id);
            Show(bv.h_aabb);
        }
    }
}