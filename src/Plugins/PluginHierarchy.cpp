//
// Created by alex on 29.07.24.
//

#include "PluginHierarchy.h"
#include "HierarchyGui.h"
#include "Engine.h"
#include "Entity.h"
#include "Transform.h"
#include "Logger.h"
#include "Picker.h"
#include "imgui.h"
#include "EventsEntity.h"

namespace Bcg {
    void MarkTransformHierarchyDirty(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Invalid entity ID: {}", entity_id);
            return;
        }

        if (!Engine::has<Hierarchy<float>>(entity_id)) {
            Log::Error("Entity does not have a Hierarchy component: {}", entity_id);
            return;
        }

        if (Engine::has<DirtyTransformHierarchy>(entity_id)) {
            return;
        }

        Engine::State().emplace_or_replace<DirtyTransformHierarchy>(entity_id);

        auto &hierarchy = Engine::State().get<Hierarchy<float>>(entity_id);

        for (auto child: hierarchy.children) {
            if (Engine::valid(child)) {
                MarkTransformHierarchyDirty(child);
            }
        }
    }

    void PluginHierarchy::attach_child(entt::entity parent, entt::entity child) {
        if (!Engine::valid(parent)) {
            Log::Error("Parent entity is not valid: {}", parent);
            return;
        }
        if (!Engine::valid(child)) {
            Log::Error("Child entity is not valid: {}", child);
            return;
        }

        if (parent == child) {
            Log::Error("Parent and child are the same entity: parent({}), child({})", parent, child);
            return;
        }

        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy<float>>(child);

        if (Engine::valid(c_hierarchy.parent)) {
            Log::Error("Child already has a parent: child({}), child_parent({})", child, c_hierarchy.parent);
            return;
        }

        //Check if child is already a child of parent
        if (c_hierarchy.parent == parent) {
            Log::Error("Parent and child parent are the same: parent({}), child_parent({})", parent,
                       c_hierarchy.parent);
            return;
        }

        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy<float>>(parent);

        if (p_hierarchy.has_child(child)) {
            Log::Error("Parent and child are already related: parent({}), child({})", parent, child);
            return;
        }

        c_hierarchy.parent = parent;
        p_hierarchy.children.push_back(child);

        MarkTransformHierarchyDirty(child);
    }

    bool PluginHierarchy::detach_child(entt::entity parent, entt::entity child) {
        if (!Engine::valid(parent)) {
            Log::Error("Parent entity is not valid: {}", parent);
            return false;
        }

        if (!Engine::valid(child)) {
            Log::Error("Child entity is not valid: {}", child);
            return false;
        }

        if (parent == child) {
            Log::Error("Parent and child are the same entity");
            return false;
        }

        if (!Engine::has<Hierarchy<float>>(parent)) {
            Log::Error("Parent does not have a Hierarchy component");
            return false;
        }

        if (!Engine::has<Hierarchy<float>>(child)) {
            Log::Error("Child does not have a Hierarchy component");
            return false;
        }

        auto &c_hierarchy = Engine::State().get<Hierarchy<float>>(child);
        auto &p_hierarchy = Engine::State().get<Hierarchy<float>>(parent);

        auto iter = std::find(p_hierarchy.children.begin(), p_hierarchy.children.end(), child);

        if (c_hierarchy.parent != parent || iter == p_hierarchy.children.end()) {
            Log::Error("Parent and child are not related");
            return false;
        }

        c_hierarchy.parent = entt::null;  // detach the child from the parent
        p_hierarchy.children.erase(iter); // remove the child from the parent's children list


        auto &c_transform = Engine::State().get<Transform<float>>(child);
        Transform<float> rel_transform = c_transform;
        auto p_cached_transform = Engine::State().get_or_emplace<CachedParentWorldTransform>(parent);
        // Update the child's world transform to be independent of the parent
        c_transform = rel_transform * p_cached_transform.transform;

        // Remove the cached transform
        Engine::State().remove<CachedParentWorldTransform>(child);
        return true;
    }

    void PluginHierarchy::attach_overlay(entt::entity parent, entt::entity overlay) {
        if (!Engine::valid(parent) || !Engine::valid(overlay)) return;

        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return;
        }

        auto &o_hierarchy = Engine::State().get_or_emplace<Hierarchy<float>>(overlay);
        //Check if child is already a child of parent
        if (o_hierarchy.parent == parent) {
            Log::Error("Parent and overlay are already related");
            return;
        }

        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy<float>>(parent);

        assert(p_hierarchy.has_overlay(overlay));

        if (Engine::valid(o_hierarchy.parent)) {
            PluginHierarchy::detach_child(o_hierarchy.parent, overlay);
        }

        o_hierarchy.parent = parent;
        p_hierarchy.children.push_back(overlay);

        //Setup the transform of the child entity
        auto &p_transform = Engine::State().get<Transform<float>>(parent);
        auto &c_transform = Engine::State().get<Transform<float>>(overlay);

        //update the child transform to be relative to the parent
        c_transform.set_local(glm::inverse(p_transform.world()) * c_transform.world());

        // Update the child's world transform
        c_transform.set_parent_world(p_transform.world());
        PluginHierarchy::mark_transforms_dirty(overlay);
    }

    bool PluginHierarchy::detach_overlay(entt::entity parent, entt::entity overlay) {
        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return false;
        }

        if (!Engine::valid(overlay) || !Engine::has<Hierarchy<float>>(overlay)) return false;

        auto &o_hierarchy = Engine::State().get<Hierarchy<float>>(overlay);
        if (o_hierarchy.parent != parent) {
            Log::Error("Parent and overlay are not related");
            return false;
        }

        o_hierarchy.parent = entt::null;

        if (Engine::valid(parent) && Engine::has<Hierarchy<float>>(parent)) {
            auto &p_hierarchy = Engine::State().get<Hierarchy<float>>(parent);
            auto iter = std::find(p_hierarchy.overlays.begin(), p_hierarchy.overlays.end(), overlay);
            if (iter != p_hierarchy.overlays.end()) {
                p_hierarchy.overlays.erase(iter);
            } else {
                Log::Error("Parent and overlay were not related. Check how this could happend!");
            }

            // Update the child's world transform to be independent of the parent
            auto &o_transform = Engine::State().get<Transform<float>>(overlay);
            o_transform.set_local(o_transform.world());
            o_transform.set_parent_world(Matrix<float, 4, 4>(1.0f));
        } else {
            Log::Warn("Called detach overlay on entity with Hierarchy component but parent is not set.");
            return false;
        }
        return true;
    }

    void PluginHierarchy::detach_children(entt::entity parent) {
        if (!Engine::valid(parent) || !Engine::has<Hierarchy<float>>(parent)) return;
        auto &p_hierarchy = Engine::State().get<Hierarchy<float>>(parent);
        for (auto child: p_hierarchy.children) {
            auto &c_hierarchy = Engine::State().get<Hierarchy<float>>(child);
            c_hierarchy.parent = entt::null;

            // Update the child's world transform to be independent of the parent
            auto &c_transform = Engine::State().get<Transform<float>>(child);
            c_transform.set_local(c_transform.world());
            c_transform.set_parent_world(Matrix<float, 4, 4>(1.0f));
        }
        p_hierarchy.children.clear();
    }

    void PluginHierarchy::detach_overlays(entt::entity parent) {
        if (!Engine::valid(parent) || !Engine::has<Hierarchy<float>>(parent)) return;
        auto &p_hierarchy = Engine::State().get<Hierarchy<float>>(parent);
        for (auto overlay: p_hierarchy.overlays) {
            auto &o_hierarchy = Engine::State().get<Hierarchy<float>>(overlay);
            o_hierarchy.parent = entt::null;

            // Update the overlay's world transform to be independent of the parent
            auto &o_transform = Engine::State().get<Transform<float>>(overlay);
            o_transform.set_local(o_transform.world());
            o_transform.set_parent_world(Matrix<float, 4, 4>(1.0f));
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
        if (!Engine::valid(child) || !Engine::has<Hierarchy<float>>(child)) return false;
        auto &c_hierarchy = Engine::State().get<Hierarchy<float>>(child);
        auto parent = c_hierarchy.parent;
        c_hierarchy.parent = entt::null;

        if (Engine::valid(parent) && Engine::has<Hierarchy<float>>(parent)) {
            auto &p_hierarchy = Engine::State().get<Hierarchy<float>>(parent);
            auto iter = std::find(p_hierarchy.children.begin(), p_hierarchy.children.end(), child);
            if (iter != p_hierarchy.children.end()) {
                p_hierarchy.children.erase(iter);
            } else {
                Log::Error("Parent and child were not related. Check how this could happend!");
            }

            // Update the child's world transform to be independent of the parent
            auto &c_transform = Engine::State().get<Transform<float>>(child);
            c_transform.set_local(c_transform.world());
            c_transform.set_parent_world(Matrix<float, 4, 4>(1.0f));
        } else {
            Log::Warn("Called detach child on entity with Hierarchy component but parent is not set.");
            return false;
        }
        return true;
    }

    void PluginHierarchy::mark_transforms_dirty(entt::entity entity) {
        if (!Engine::valid(entity) || !Engine::has<Transform<float>>(entity)) return;

        auto &p_transform = Engine::State().get<Transform<float>>(entity);
        p_transform.dirty = true;

        if (!Engine::has<Hierarchy<float>>(entity)) return;

        auto &p_hierarchy = Engine::State().get<Hierarchy<float>>(entity);
        for (auto child: p_hierarchy.children) {
            auto &c_transform = Engine::State().get<Transform<float>>(child);
            if (c_transform.dirty) continue;
            PluginHierarchy::mark_transforms_dirty(child);
        }
    }


    void PluginHierarchy::update_transforms(entt::entity entity) {
        if (!Engine::valid(entity) || !Engine::has<Hierarchy<float>>(entity)) return;
        auto &e_transform = Engine::State().get<Transform<float>>(entity);
        if (!e_transform.dirty) return;

        if (!Engine::has<Hierarchy<float>>(entity)) {
            e_transform.set_parent_world(Matrix<float, 4, 4>(1.0f));
        } else {
            auto &e_hierarchy = Engine::State().get<Hierarchy<float>>(entity);
            if (Engine::valid(e_hierarchy.parent)) {
                auto &p_transform = Engine::State().get<Transform>(e_hierarchy.parent);
                e_transform.set_parent_world(p_transform.world());
            } else {
                e_transform.set_parent_world(Matrix<float, 4, 4>(1.0f));
            }

            for (auto child: e_hierarchy.children) {
                PluginHierarchy::update_transforms(child);
            }
        }
        e_transform.dirty = false;
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

    namespace Commands {
        void Setup<Hierarchy<float>>::execute() const {
            if (!Engine::valid(entity_id)) {
                return;
            }

            if (!Engine::has<Hierarchy<float>>(entity_id)) {
                Engine::State().emplace<Hierarchy<float>>(entity_id);
            }

            Log::Info("{} for entity {}", name, entity_id);
        }

        void Cleanup<Hierarchy<float>>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            if (!Engine::has<Hierarchy<float>>(entity_id)) {
                Log::Warn(name + "Entity does not have a PointCloud. Abort Command");
                return;
            }

            auto &hierarchy = Engine::require<Hierarchy<float>>(entity_id);
            PluginHierarchy::detach_child(hierarchy.parent, entity_id);
            PluginHierarchy::detach_children(entity_id);

            Engine::Dispatcher().trigger(Events::Entity::PreRemove<Hierarchy<float>>{entity_id});
            Engine::State().remove<Hierarchy<float>>(entity_id);
            Engine::Dispatcher().trigger(Events::Entity::PostRemove<Hierarchy<float>>{entity_id});
            Log::Info("{} for entity {}", name, entity_id);
        }

        void UpdateTransformsDeferred::execute() const {
            PluginHierarchy::update_transforms(entity_id);
        }
    }
}