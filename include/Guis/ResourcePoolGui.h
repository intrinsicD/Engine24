//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_RESOURCEPOOLGUI_H
#define ENGINE24_RESOURCEPOOLGUI_H

#include "ResourcePool.h"
#include "imgui.h"

namespace Bcg{
    template<typename T>
    void ShowGui(const char *label, ResourcePool<T> &pool) {
        ImGui::Text("Resource Pool: %s", label);
        ImGui::Text("Capacity: %d", pool.capacity());
        ImGui::Text("Size: %d", pool.size());
        ImGui::Text("Size Active: %d", pool.size_active());
        ImGui::Text("Size Free: %d", pool.size_free());

        if (ImGui::TreeNode("Resources")) {
            for (auto &resource : pool) {
                ImGui::Text("Resource: %d", resource.index());
                ImGui::Text("Deleted: %s", resource.is_deleted() ? "true" : "false");
            }
            ImGui::TreePop();
        }
    }
}

#endif //ENGINE24_RESOURCEPOOLGUI_H
