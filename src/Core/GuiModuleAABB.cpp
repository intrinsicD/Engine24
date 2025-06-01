//
// Created by alex on 25.11.24.
//

#include "GuiModuleAABB.h"
#include "imgui.h"
#include "Engine.h"
#include "Picker.h"
#include "AABBGui.h"
#include "PropertiesGui.h"

namespace Bcg {
    void GuiModuleAABB::activate()  {
        if (base_activate()) {

        }
    }

    void GuiModuleAABB::deactivate()  {
        if (base_deactivate()) {

        }
    }

    static bool show_gui = false;
    static bool show_pool_gui = false;

    void GuiModuleAABB::render_menu()  {
        if (ImGui::BeginMenu("Entity")) {
            if (ImGui::BeginMenu("AABB")) {
                ImGui::MenuItem("Instance", nullptr, &show_gui);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Pools")) {
            ImGui::MenuItem("AABBs", nullptr, &show_pool_gui);
            ImGui::EndMenu();
        }
    }

    void GuiModuleAABB::render_gui()  {
        if (show_gui) {
            if (ImGui::Begin("AABB", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                show(picked.entity.id);
            }
            ImGui::End();
        }
        if (show_pool_gui) {
            if (ImGui::Begin("AABBs", &show_pool_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &pool = Engine::Context().get<Pool<AABB> >();
                show(pool);
            }
            ImGui::End();
        }
    }

    void GuiModuleAABB::show(const PoolHandle<AABB> &h_aabb) {
        Gui::Show(h_aabb);
    }

    void GuiModuleAABB::show(const AABB &aabb) {
        Gui::Show(aabb);
    }

    void GuiModuleAABB::show(Pool<AABB> &pool) {
        Gui::Show(*pool.ref_count.base());
        Gui::Show(*pool.objects.base());
        if (ImGui::CollapsingHeader("Properties")) {
            Gui::Show("#AABB Pool", pool.properties);
        }
    }

    void GuiModuleAABB::show(entt::entity entity_id) {
        Gui::Show(entity_id);
    }
}
