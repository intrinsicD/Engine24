//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_ENTITYSELECTION_H
#define ENGINE24_ENTITYSELECTION_H

#include "entt/entity/entity.hpp"

namespace Bcg {
    class EntitySelection {
    public:
        EntitySelection() = default;

        ~EntitySelection() = default;

        void select_entity(entt::entity entity_id) {
            selected_entity = entity_id;
        }

        void deselect_entity() {
            selected_entity = entt::null;
        }

        bool is_entity_selected(entt::entity entity_id) const{
            return selected_entity == entity_id;
        }

        entt::entity get_selected_entity() const{
            return selected_entity;
        }

    private:
        entt::entity selected_entity;
    };
}

#endif //ENGINE24_ENTITYSELECTION_H
