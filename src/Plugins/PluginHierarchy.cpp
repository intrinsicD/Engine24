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
#include "Commands/Command.h"

namespace Bcg {
    namespace Commands {
        struct MakeTransfromLocalToParent : public AbstractCommand {
            explicit MakeTransfromLocalToParent(entt::entity entity_id) : AbstractCommand("MakeTransfromLocalToParent"),
                                                                          entity_id(entity_id) {
            }

            void execute() const override {
                if (!Engine::valid(entity_id)) {
                    Log::Error("Entity is not valid: {}", entity_id);
                    return;
                }
                if (Engine::has<CachedWorldTransform<float>>(entity_id)) {
                    // The transform is already local
                    return;
                }
                if (!Engine::has<Hierarchy<float> >(entity_id)) {
                    Log::Error("Entity does not have a Hierarchy component: {}", entity_id);
                    return;
                }
                if (!Engine::has<Transform<float> >(entity_id)) {
                    Log::Error("Entity does not have a Transform component: {}", entity_id);
                    return;
                }

                auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity_id);


                auto &transform = Engine::State().get<Transform<float> >(entity_id);
                Engine::State().emplace<CachedWorldTransform<float> >(entity_id, transform);
                // world = rel * parent_world
                // world * parent_world_inv = rel
                // if parent is not valid, then we are at the root and transform is already "local".
                if (Engine::valid(hierarchy.parent)) {
                    //Assume parent world is cached and not dirty
                    auto &p_cached = Engine::State().get<CachedWorldTransform<float> >(hierarchy.parent);
                    transform = p_cached.transform.inverse() * transform;
                }
            }

            entt::entity entity_id;
        };

        struct UpdateCachedWorldTransfrom : public AbstractCommand {
            explicit UpdateCachedWorldTransfrom(entt::entity entity_id) : AbstractCommand("UpdateCachedWorldTransfrom"),
                                                                          entity_id(entity_id) {
            }

            void execute() const override {
                if (!Engine::valid(entity_id)) {
                    Log::Error("Entity is not valid: {}", entity_id);
                    return;
                }
                if (!Engine::has<DirtyTransformHierarchy>(entity_id)) {
                    return;
                }
                if (!Engine::has<Transform<float> >(entity_id)) {
                    Log::Error("Entity does not have a Transform component: {}", entity_id);
                    return;
                }
                if (!Engine::has<Hierarchy<float> >(entity_id)) {
                    Log::Error("Entity does not have a Hierarchy component: {}", entity_id);
                    return;
                }

                auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity_id);
                auto &transform = Engine::State().get<Transform<float> >(entity_id);
                if (!Engine::valid(hierarchy.parent)) {
                    Engine::State().emplace_or_replace<CachedWorldTransform<float> >(entity_id, transform);
                } else {
                    auto &p_cached = Engine::State().get<CachedWorldTransform<float> >(hierarchy.parent);
                    Engine::State().emplace_or_replace<CachedWorldTransform<float> >(
                        entity_id, p_cached.transform * transform);
                }

                Engine::State().remove<DirtyTransformHierarchy>(entity_id);
            }

            entt::entity entity_id;
        };

        struct MakeTransfromGlobal : public AbstractCommand {
            explicit MakeTransfromGlobal(entt::entity entity_id) : AbstractCommand("MakeTransfromGlobal"),
                                                                   entity_id(entity_id) {
            }

            void execute() const override {
                if (!Engine::valid(entity_id)) {
                    Log::Error("Entity is not valid: {}", entity_id);
                    return;
                }
                if (!Engine::has<CachedWorldTransform<float>>(entity_id)) {
                    Log::Error("Entity does not have a CachedWorldTransform component: {}", entity_id);
                    return;
                }
                if (!Engine::has<Transform<float> >(entity_id)) {
                    Log::Error("Entity does not have a Transform component: {}", entity_id);
                    return;
                }
                auto &transform = Engine::State().get<Transform<float> >(entity_id);

                transform = Engine::State().get<CachedWorldTransform<float> >(entity_id).transform;
                Engine::State().remove<CachedWorldTransform<float> >(entity_id);
            }

            entt::entity entity_id;
        };
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

        auto &c_hierarchy = Engine::State().get_or_emplace<Hierarchy<float> >(child);

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

        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy<float> >(parent);

        if (p_hierarchy.has_child(child)) {
            Log::Error("Parent and child are already related: parent({}), child({})", parent, child);
            return;
        }

        c_hierarchy.parent = parent;
        p_hierarchy.children.push_back(child);

        mark_transforms_dirty(child, true);
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

        if (!Engine::has<Hierarchy<float> >(parent)) {
            Log::Error("Parent does not have a Hierarchy component");
            return false;
        }

        if (!Engine::has<Hierarchy<float> >(child)) {
            Log::Error("Child does not have a Hierarchy component");
            return false;
        }

        auto &c_hierarchy = Engine::State().get<Hierarchy<float> >(child);
        auto &p_hierarchy = Engine::State().get<Hierarchy<float> >(parent);

        auto iter = std::find(p_hierarchy.children.begin(), p_hierarchy.children.end(), child);

        if (c_hierarchy.parent != parent || iter == p_hierarchy.children.end()) {
            Log::Error("Parent and child are not related");
            return false;
        }

        c_hierarchy.parent = entt::null; // detach the child from the parent
        p_hierarchy.children.erase(iter); // remove the child from the parent's children list

        auto &c_transform = Engine::State().get<Transform<float> >(child);
        Transform<float> rel_transform = c_transform;
        auto p_cached_transform = Engine::State().get_or_emplace<CachedWorldTransform<float>>(parent);
        // Update the child's world transform to be independent of the parent
        c_transform = rel_transform * p_cached_transform.transform;

        // Remove the cached transform
        Engine::State().remove<CachedWorldTransform<float>>(child);
        return true;
    }

    void PluginHierarchy::attach_overlay(entt::entity parent, entt::entity overlay) {
        if (!Engine::valid(parent)) {
            Log::Error("Parent entity is not valid: {}", parent);
            return;
        }
        if (!Engine::valid(overlay)) {
            Log::Error("Overlay entity is not valid: {}", overlay);
            return;
        }

        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity: parent({}), overlay({})", parent, overlay);
            return;
        }

        auto &o_hierarchy = Engine::State().get_or_emplace<Hierarchy<float> >(overlay);

        if (Engine::valid(o_hierarchy.parent)) {
            Log::Error("Overlay already has a parent: child({}), overlay_parent({})", overlay, o_hierarchy.parent);
            return;
        }

        //Check if child is already a child of parent
        if (o_hierarchy.parent == parent) {
            Log::Error("Parent and overlay parent are the same: parent({}), overlay_parent({})", parent,
                       o_hierarchy.parent);
            return;
        }

        auto &p_hierarchy = Engine::State().get_or_emplace<Hierarchy<float> >(parent);

        if (p_hierarchy.has_child(overlay)) {
            Log::Error("Parent and overlay are already related: parent({}), overlay({})", parent, overlay);
            return;
        }

        o_hierarchy.parent = parent;
        p_hierarchy.overlays.push_back(overlay);

        mark_transforms_dirty(overlay, false);
    }

    bool PluginHierarchy::detach_overlay(entt::entity parent, entt::entity overlay) {
        if (!Engine::valid(parent)) {
            Log::Error("Parent entity is not valid: {}", parent);
            return false;
        }

        if (!Engine::valid(overlay)) {
            Log::Error("Overlay entity is not valid: {}", overlay);
            return false;
        }

        if (parent == overlay) {
            Log::Error("Parent and overlay are the same entity");
            return false;
        }

        if (!Engine::has<Hierarchy<float> >(parent)) {
            Log::Error("Parent does not have a Hierarchy component");
            return false;
        }

        if (!Engine::has<Hierarchy<float> >(overlay)) {
            Log::Error("Overlay does not have a Hierarchy component");
            return false;
        }

        auto &o_hierarchy = Engine::State().get<Hierarchy<float> >(overlay);
        auto &p_hierarchy = Engine::State().get<Hierarchy<float> >(parent);

        auto iter = std::find(p_hierarchy.overlays.begin(), p_hierarchy.overlays.end(), overlay);

        if (o_hierarchy.parent != parent || iter == p_hierarchy.overlays.end()) {
            Log::Error("Parent and overlay are not related");
            return false;
        }

        o_hierarchy.parent = entt::null; // detach the overlay from the parent
        p_hierarchy.overlays.erase(iter); // remove the overlay from the parent's overlays list

        auto &o_transform = Engine::State().get<Transform<float> >(overlay);
        Transform<float> rel_transform = o_transform;
        auto p_cached_transform = Engine::State().get_or_emplace<CachedWorldTransform<float>>(parent);
        // Update the child's world transform to be independent of the parent
        o_transform = rel_transform * p_cached_transform.transform;

        // Remove the cached transform
        Engine::State().remove<CachedWorldTransform<float>>(overlay);
        return true;
    }

    void PluginHierarchy::detach_children(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Entity is not valid: {}", entity_id);
            return;
        }

        if (!Engine::has<Hierarchy<float> >(entity_id)) {
            Log::Error("Entity does not have a Hierarchy component: {}", entity_id);
            return;
        }

        auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity_id);

        for (auto child: hierarchy.children) {
            detach_child(entity_id, child);
        }

        hierarchy.children.clear();
    }

    void PluginHierarchy::detach_overlays(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Entity is not valid: {}", entity_id);
            return;
        }

        if (!Engine::has<Hierarchy<float> >(entity_id)) {
            Log::Error("Entity does not have a Hierarchy component: {}", entity_id);
            return;
        }

        auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity_id);

        for (auto child: hierarchy.overlays) {
            detach_overlay(entity_id, child);
        }

        hierarchy.overlays.clear();
    }

    void PluginHierarchy::detach_all(entt::entity parent) {
        detach_children(parent);
        detach_overlays(parent);
    }

    void PluginHierarchy::attach_parent(entt::entity child, entt::entity new_parent) {
        attach_child(new_parent, child);
    }

    bool PluginHierarchy::detach_parent(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Child entity is not valid: {}", entity_id);
            return false;
        }

        if (!Engine::has<Hierarchy<float> >(entity_id)) {
            Log::Error("Child does not have a Hierarchy component: {}", entity_id);
            return false;
        }

        auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity_id);

        return detach_child(hierarchy.parent, entity_id);
    }

    void PluginHierarchy::mark_transforms_dirty(entt::entity entity, bool children_or_overlays) {
        if (!Engine::valid(entity)) {
            Log::Error("Invalid entity ID: {}", entity);
            return;
        }

        if (!Engine::has<Hierarchy<float> >(entity)) {
            Log::Error("Entity does not have a Hierarchy component: {}", entity);
            return;
        }

        if (Engine::has<DirtyTransformHierarchy>(entity)) return;

        Engine::State().emplace<DirtyTransformHierarchy>(entity);

        auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity);
        if (children_or_overlays) {
            for (auto child: hierarchy.children) {
                mark_transforms_dirty(child, children_or_overlays);
            }
        } else {
            for (auto child: hierarchy.overlays) {
                mark_transforms_dirty(child, children_or_overlays);
            }
        }
    }


    void PluginHierarchy::update_transforms(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Invalid entity ID: {}", entity_id);
            return;
        }
        if (!Engine::has<Hierarchy<float> >(entity_id)) {
            Log::Error("Entity does not have a Hierarchy component: {}", entity_id);
            return;
        }
        auto &hierarchy = Engine::State().get<Hierarchy<float> >(entity_id);
        if (Engine::valid(hierarchy.parent)) {
            if (!Engine::has<DirtyTransformHierarchy>(hierarchy.parent)) {
                auto p_cached = Engine::State().get<CachedWorldTransform<float> >(hierarchy.parent);
                //marks whether the parents transform is local or not
                auto &transform = Engine::State().get<Transform<float> >(entity_id); //always exists

                if (!Engine::has<CachedWorldTransform<float> >(entity_id)) {
                    //make transform relative to parent
                    transform = p_cached.transform.inverse() * transform;
                }

                //update the cached world Transform
                auto cached = Engine::State().get_or_emplace<CachedWorldTransform<float> >(entity_id);
                cached.transform = p_cached.transform * transform;
            }
        }

        Engine::State().remove<DirtyTransformHierarchy>(entity_id);

        for (auto child: hierarchy.children) {
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

    namespace Commands {
        void Setup<Hierarchy<float> >::execute() const {
            if (!Engine::valid(entity_id)) {
                return;
            }

            if (!Engine::has<Hierarchy<float> >(entity_id)) {
                Engine::State().emplace<Hierarchy<float> >(entity_id);
            }

            Log::Info("{} for entity {}", name, entity_id);
        }

        void Cleanup<Hierarchy<float> >::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            if (!Engine::has<Hierarchy<float> >(entity_id)) {
                Log::Warn(name + "Entity does not have a Hierarchy. Abort Command");
                return;
            }

            auto &hierarchy = Engine::require<Hierarchy<float> >(entity_id);
            PluginHierarchy::detach_child(hierarchy.parent, entity_id);
            PluginHierarchy::detach_children(entity_id);

            Engine::Dispatcher().trigger(Events::Entity::PreRemove<Hierarchy<float> >{entity_id});
            Engine::State().remove<Hierarchy<float> >(entity_id);
            Engine::Dispatcher().trigger(Events::Entity::PostRemove<Hierarchy<float> >{entity_id});
            Log::Info("{} for entity {}", name, entity_id);
        }

        void UpdateTransformsDeferred::execute() const {
            auto view = Engine::State().view<Hierarchy<float>, Transform<float>, DirtyTransformHierarchy>();
            for (auto entity_id: view) {
                auto &hierarchy = view.get<Hierarchy<float> >(entity_id);

                if (!Engine::valid(hierarchy.parent) || !Engine::has<DirtyTransformHierarchy>(hierarchy.parent)) {
                    PluginHierarchy::update_transforms(entity_id);
                    Engine::State().remove<DirtyTransformHierarchy>(entity_id);
                }
            }
        }
    }
}
