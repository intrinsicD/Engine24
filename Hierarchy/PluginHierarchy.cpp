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
    void PluginHierarchy::add_child(entt::entity parent, entt::entity child) {
        if (parent == child) {
            Log::Error("Parent and child are the same entity");
            return;
        }
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy>(child);
        c_hierarchy.parent = parent;
        p_hierarchy.children.push_back(child);

        //Setup the transform of the child entity
        auto &p_transform = Engine::State().get_or_emplace<Transform>(parent);
        auto &c_transform = Engine::State().get_or_emplace<Transform>(child);

        //update the child transform to be relative to the parent
        c_transform.local = p_transform.world.inverse() * c_transform.world; // Compute local transform
        c_transform.dirty = true;

        // Update the child's world transform
        c_transform.update(p_transform.world.matrix());
    }

    bool PluginHierarchy::remove_child(entt::entity parent, entt::entity child) {
        if (parent == child) {
            Log::Error("Parent and child are the same entity");
            return false;
        }
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy>(child);
        auto iter = std::find(p_hierarchy.children.begin(), p_hierarchy.children.end(), child);
        if (iter != p_hierarchy.children.end()) {
            c_hierarchy.parent = entt::null;
            p_hierarchy.children.erase(iter);

            auto &c_transform = Engine::State().get_or_emplace<Transform>(child);
            c_transform.local = c_transform.world; // Detach from parent
            c_transform.dirty = true;
            return true;
        }
        return false;
    }

    void PluginHierarchy::add_overlay(entt::entity parent, entt::entity overlay) {
        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return;
        }
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        auto &o_hierarchy = Engine::State().get_or_emplace<Hierarchy>(overlay);
        o_hierarchy.parent = parent;
        p_hierarchy.overlays.push_back(overlay);

        // Setup the transform of the overlay entity
        auto &p_transform = Engine::State().get_or_emplace<Transform>(parent);
        auto &o_transform = Engine::State().get_or_emplace<Transform>(overlay);
        o_transform.update(p_transform.world.matrix());
        o_transform.dirty = true;
    }

    void PluginHierarchy::remove_overlay(entt::entity parent, entt::entity overlay) {
        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return;
        }
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        auto &o_hierarchy = Engine::State().get_or_emplace<Hierarchy>(overlay);
        auto iter = std::find(p_hierarchy.overlays.begin(), p_hierarchy.overlays.end(), overlay);
        if (iter != p_hierarchy.overlays.end()) {
            o_hierarchy.parent = entt::null;
            p_hierarchy.overlays.erase(iter);

            // Update the overlay's world transform to be independent of the parent
            auto &o_transform = Engine::State().get_or_emplace<Transform>(overlay);
            o_transform.local = o_transform.world; // Detach from parent
            o_transform.dirty = true;
        }
    }

    void PluginHierarchy::clear_children(entt::entity parent) {
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        for (auto child: p_hierarchy.children) {
            auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy>(child);
            c_hierarchy.parent = entt::null;

            // Update the child's world transform to be independent of the parent
            auto &c_transform = Engine::State().get_or_emplace<Transform>(child);
            c_transform.local = c_transform.world; // Detach from parent
            c_transform.dirty = true;
        }
        p_hierarchy.children.clear();
    }

    void PluginHierarchy::clear_overlays(entt::entity parent) {
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        for (auto overlay: p_hierarchy.overlays) {
            auto &o_hierarchy = Engine::State().get_or_emplace<Hierarchy>(overlay);
            o_hierarchy.parent = entt::null;

            // Update the overlay's world transform to be independent of the parent
            auto &o_transform = Engine::State().get_or_emplace<Transform>(overlay);
            o_transform.local = o_transform.world; // Detach from parent
            o_transform.dirty = true;
        }
        p_hierarchy.overlays.clear();
    }

    void PluginHierarchy::clear(entt::entity parent) {
        clear_children(parent);
        clear_overlays(parent);
    }

    void PluginHierarchy::set_parent(entt::entity child, entt::entity new_parent) {
        if (child == new_parent) {
            Log::Error("Parent and child are the same entity");
            return;
        }
        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy>(child);
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(new_parent);
        if (c_hierarchy.parent != entt::null) {
            auto &old_parent_hierarchy = Engine::State().get_or_emplace<Hierarchy>(c_hierarchy.parent);
            auto iter = std::find(old_parent_hierarchy.children.begin(), old_parent_hierarchy.children.end(), child);
            if (iter != old_parent_hierarchy.children.end()) {
                old_parent_hierarchy.children.erase(iter);
            }
        }
        c_hierarchy.parent = new_parent;
        p_hierarchy.children.push_back(child);

        // Update the child's transform to be relative to the new parent
        auto &p_transform = Engine::State().get_or_emplace<Transform>(new_parent);
        auto &c_transform = Engine::State().get_or_emplace<Transform>(child);
        c_transform.update(p_transform.world.matrix());
        c_transform.dirty = true;
    }

    void PluginHierarchy::remove_parent(entt::entity child) {
        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy>(child);
        if (c_hierarchy.parent != entt::null) {
            auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(c_hierarchy.parent);
            auto iter = std::find(p_hierarchy.children.begin(), p_hierarchy.children.end(), child);
            if (iter != p_hierarchy.children.end()) {
                p_hierarchy.children.erase(iter);
            }
            c_hierarchy.parent = entt::null;

            // Update the child's world transform to be independent of the parent
            auto &c_transform = Engine::State().get_or_emplace<Transform>(child);
            c_transform.local = c_transform.world; // Detach from parent
            c_transform.dirty = true;
        }
    }


    void PluginHierarchy::update_transforms(entt::entity parent) {
        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy>(parent);
        auto &p_transform = Engine::State().get_or_emplace<Transform>(parent);

        for (auto child: p_hierarchy.children) {
            auto &c_transform = Engine::State().get_or_emplace<Transform>(child);
            c_transform.update(p_transform.world.matrix());

            // Recursively update children's children
            update_transforms(child);
        }
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
}