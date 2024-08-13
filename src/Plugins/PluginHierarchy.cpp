//
// Created by alex on 29.07.24.
//

#include "PluginHierarchy.h"
#include "HierarchyGui.h"
#include "Engine.h"
#include "Transform.h"
#include "Logger.h"
#include "Picker.h"
#include "imgui.h"

namespace Bcg {
    void PluginHierarchy::attach_child(entt::entity parent, entt::entity child) {
        if (!Engine::valid(parent) || !Engine::valid(child)) return;

        if (parent == child) {
            Log::Error("Parent and child are the same entity");
            return;
        }

        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy>(child);
        //Check if child is already a child of parent
        if (c_hierarchy.parent == parent) {
            Log::Error("Parent and child are already related");
            return;
        }

        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);

        assert(!p_hierarchy.has_child(child));

        if (Engine::valid(c_hierarchy.parent)) {
            //TODO what happens if child already has a parent which is different from parent?
            PluginHierarchy::detach_child(c_hierarchy.parent, child);
        }

        c_hierarchy.parent = parent;
        p_hierarchy.children.push_back(child);

        //Setup the transform of the child entity
        auto &p_transform = Engine::State().get<Transform>(parent);
        auto &c_transform = Engine::State().get<Transform>(child);

        if (false) {
            //old way, changed the local transform resulted in repositioned guizmo

            //update the child transform to be relative to the parent
            c_transform.set_local(p_transform.world().inverse() * c_transform.world().matrix());

            // Update the child's world transform
            c_transform.update_world(p_transform.world());
        } else {
            //update the child transform to be relative to the parent
            c_transform.set_local(p_transform.world().inverse() * c_transform.world().matrix());

            // Update the child's world transform
            c_transform.update_world(p_transform.world());
        }


        PluginHierarchy::mark_transforms_dirty(child);
    }

    bool PluginHierarchy::detach_child(entt::entity parent, entt::entity child) {
        if (parent == child) {
            Log::Error("Parent and child are the same entity");
            return false;
        }

        if (!Engine::valid(child) || !Engine::has<Hierarchy>(child)) return false;

        auto &c_hierarchy = Engine::State().get<Hierarchy>(child);
        if (c_hierarchy.parent != parent) {
            Log::Error("Parent and child are not related");
            return false;
        }

        return detach_parent(child);
    }

    void PluginHierarchy::attach_overlay(entt::entity parent, entt::entity overlay) {
        if (!Engine::valid(parent) || !Engine::valid(overlay)) return;

        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return;
        }

        auto &o_hierarchy = Engine::State().get_or_emplace<Hierarchy>(overlay);
        //Check if child is already a child of parent
        if (o_hierarchy.parent == parent) {
            Log::Error("Parent and overlay are already related");
            return;
        }

        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);

        assert(p_hierarchy.has_overlay(overlay));

        if (Engine::valid(o_hierarchy.parent)) {
            PluginHierarchy::detach_child(o_hierarchy.parent, overlay);
        }

        o_hierarchy.parent = parent;
        p_hierarchy.children.push_back(overlay);

        //Setup the transform of the child entity
        auto &p_transform = Engine::State().get<Transform>(parent);
        auto &c_transform = Engine::State().get<Transform>(overlay);

        //update the child transform to be relative to the parent
        c_transform.set_local(p_transform.world().inverse() * c_transform.world().matrix());

        // Update the child's world transform
        c_transform.update_world(p_transform.world());
        PluginHierarchy::mark_transforms_dirty(overlay);
    }

    bool PluginHierarchy::detach_overlay(entt::entity parent, entt::entity overlay) {
        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return false;
        }

        if (!Engine::valid(overlay) || !Engine::has<Hierarchy>(overlay)) return false;

        auto &o_hierarchy = Engine::State().get<Hierarchy>(overlay);
        if (o_hierarchy.parent != parent) {
            Log::Error("Parent and overlay are not related");
            return false;
        }

        o_hierarchy.parent = entt::null;

        if (Engine::valid(parent) && Engine::has<Hierarchy>(parent)) {
            auto &p_hierarchy = Engine::State().get<Hierarchy>(parent);
            auto iter = std::find(p_hierarchy.overlays.begin(), p_hierarchy.overlays.end(), overlay);
            if (iter != p_hierarchy.overlays.end()) {
                p_hierarchy.overlays.erase(iter);
            } else {
                Log::Error("Parent and overlay were not related. Check how this could happend!");
            }

            // Update the child's world transform to be independent of the parent
            auto &o_transform = Engine::State().get<Transform>(overlay);
            o_transform.set_local(o_transform.world());
            o_transform.update_world(Matrix<float, 4, 4>::Identity());
        } else {
            Log::Warn("Called detach overlay on entity with Hierarchy component but parent is not set.");
            return false;
        }
        return true;
    }

    void PluginHierarchy::detach_children(entt::entity parent) {
        if (!Engine::valid(parent) || !Engine::has<Hierarchy>(parent)) return;
        auto &p_hierarchy = Engine::State().get<Hierarchy>(parent);
        for (auto child: p_hierarchy.children) {
            auto &c_hierarchy = Engine::State().get<Hierarchy>(child);
            c_hierarchy.parent = entt::null;

            // Update the child's world transform to be independent of the parent
            auto &c_transform = Engine::State().get<Transform>(child);
            c_transform.set_local(c_transform.world());
            c_transform.update_world(Matrix<float, 4, 4>::Identity());
        }
        p_hierarchy.children.clear();
    }

    void PluginHierarchy::detach_overlays(entt::entity parent) {
        if (!Engine::valid(parent) || !Engine::has<Hierarchy>(parent)) return;
        auto &p_hierarchy = Engine::State().get<Hierarchy>(parent);
        for (auto overlay: p_hierarchy.overlays) {
            auto &o_hierarchy = Engine::State().get<Hierarchy>(overlay);
            o_hierarchy.parent = entt::null;

            // Update the overlay's world transform to be independent of the parent
            auto &o_transform = Engine::State().get<Transform>(overlay);
            o_transform.set_local(o_transform.world());
            o_transform.update_world(Matrix<float, 4, 4>::Identity());
        }
        p_hierarchy.overlays.clear();
    }

    void PluginHierarchy::detach_all(entt::entity parent) {
        detach_children(parent);
        detach_overlays(parent);
    }

    void PluginHierarchy::attach_parent(entt::entity child, entt::entity new_parent) {
        attach_child(new_parent, child);
    }

    bool PluginHierarchy::detach_parent(entt::entity child) {
        if (!Engine::valid(child) || !Engine::has<Hierarchy>(child)) return false;
        auto &c_hierarchy = Engine::State().get<Hierarchy>(child);
        auto parent = c_hierarchy.parent;
        c_hierarchy.parent = entt::null;

        if (Engine::valid(parent) && Engine::has<Hierarchy>(parent)) {
            auto &p_hierarchy = Engine::State().get<Hierarchy>(parent);
            auto iter = std::find(p_hierarchy.children.begin(), p_hierarchy.children.end(), child);
            if (iter != p_hierarchy.children.end()) {
                p_hierarchy.children.erase(iter);
            } else {
                Log::Error("Parent and child were not related. Check how this could happend!");
            }

            // Update the child's world transform to be independent of the parent
            auto &c_transform = Engine::State().get<Transform>(child);
            c_transform.set_local(c_transform.world());
            c_transform.update_world(Matrix<float, 4, 4>::Identity());
        } else {
            Log::Warn("Called detach child on entity with Hierarchy component but parent is not set.");
            return false;
        }
        return true;
    }

    void PluginHierarchy::mark_transforms_dirty(entt::entity entity) {
        if (!Engine::valid(entity) || !Engine::has<Transform>(entity)) return;

        auto &p_transform = Engine::State().get<Transform>(entity);
        p_transform.mark_dirty();

        if (!Engine::has<Hierarchy>(entity)) return;

        auto &p_hierarchy = Engine::State().get<Hierarchy>(entity);
        for (auto child: p_hierarchy.children) {
            auto &c_transform = Engine::State().get<Transform>(child);
            if (c_transform.is_dirty()) continue;
            PluginHierarchy::mark_transforms_dirty(child);
        }
    }


    void PluginHierarchy::update_transforms(entt::entity entity) {
        if (!Engine::valid(entity) || !Engine::has<Hierarchy>(entity)) return;
        auto &e_transform = Engine::State().get<Transform>(entity);
        if (!e_transform.is_dirty()) return;

        if (!Engine::has<Hierarchy>(entity)) {
            e_transform.update_world(Matrix<float, 4, 4>::Identity());
        } else {
            auto &e_hierarchy = Engine::State().get<Hierarchy>(entity);
            if (Engine::valid(e_hierarchy.parent)) {
                auto &p_transform = Engine::State().get<Transform>(e_hierarchy.parent);
                e_transform.update_world(p_transform.world());
            } else {
                e_transform.update_world(Matrix<float, 4, 4>::Identity());
            }

            for (auto child: e_hierarchy.children) {
                PluginHierarchy::update_transforms(child);
            }
        }
        e_transform.mark_clean();
    }

    void PluginHierarchy::activate() {
        Plugin::activate();
    }

    void PluginHierarchy::begin_frame() {

    }

    void PluginHierarchy::update() {

    }

    void PluginHierarchy::end_frame() {

    }

    void PluginHierarchy::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginHierarchy::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginHierarchy::render_gui() {
        if (show_gui) {
            if (ImGui::Begin("Hierarchy", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                if (Engine::valid(picked.entity.id)) {
                    Gui::ShowHierarchy(picked.entity.id);
                }
            }
            ImGui::End();
        }
    }

    void PluginHierarchy::render() {

    }

    namespace Commands{
        void UpdateTransformsDeferred::execute() const {
            PluginHierarchy::update_transforms(entity_id);
        }
    }
}