//
// Created by alex on 28.07.24.
//

#include "AABBGui.h"
#include "Engine.h"
#include "imgui.h"

namespace Bcg::Gui {
    void Show(const AABB &aabb) {
        ImGui::Text("Min: (%f, %f, %f)", aabb.min.x, aabb.min.y, aabb.min.z);
        ImGui::Text("Max: (%f, %f, %f)", aabb.max.x, aabb.max.y, aabb.max.z);
    }


    void Show(entt::entity entity_id) {
        if (Engine::State().all_of<AABB>(entity_id)) {
            auto &aabb_handle = Engine::State().get<AABB>(entity_id);
            Show(aabb_handle);
            return;
        } else if (Engine::State().all_of<AABB>(entity_id)) {
            auto &aabb = Engine::State().get<AABB>(entity_id);
            Show(aabb);
            return;
        }
    }
}