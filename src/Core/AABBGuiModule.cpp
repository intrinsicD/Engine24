//
// Created by alex on 25.11.24.
//

#include "AABBGuiModule.h"
#include "imgui.h"
#include "Engine.h"
#include "Picker.h"
#include "AABBGui.h"
#include "PropertiesGui.h"

namespace Bcg {
    void AABBGuiModule::activate()  {

    }

    void AABBGuiModule::deactivate()  {
    }

    static bool show_gui = false;
    static bool show_pool_gui = false;

    void AABBGuiModule::render_menu()  {
        if (ImGui::BeginMenu("Entity")) {
            if (ImGui::BeginMenu("AABB")) {
                ImGui::MenuItem("Instance", nullptr, &show_gui);
                ImGui::MenuItem("Pool", nullptr, &show_pool_gui);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void AABBGuiModule::render_gui()  {
        if (show_gui) {
            if (ImGui::Begin("AABB", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                render(picked.entity.id);
            }
            ImGui::End();
        }
        if (show_pool_gui) {
            if (ImGui::Begin("AABB Pool", &show_pool_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &pool = Engine::Context().get<Pool<AABB> >();
                render(pool);
            }
            ImGui::End();
        }
    }

    void AABBGuiModule::render(const PoolHandle<AABB> &h_aabb) {
        Gui::Show(h_aabb);
    }

    void AABBGuiModule::render(const AABB &aabb) {
        Gui::Show(aabb);
    }

    void AABBGuiModule::render(Pool<AABB> &pool) {
        Gui::Show(*pool.ref_count.base());
        Gui::Show(*pool.objects.base());
        if (ImGui::CollapsingHeader("Properties")) {
            Gui::Show("#AABB Pool", pool.properties);
        }
    }

    void AABBGuiModule::render(entt::entity entity_id) {
        Gui::Show(entity_id);
    }
}
