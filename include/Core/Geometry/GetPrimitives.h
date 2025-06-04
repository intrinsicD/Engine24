//
// Created by alex on 01.08.24.
//

#ifndef ENGINE24_GETPRIMITIVES_H
#define ENGINE24_GETPRIMITIVES_H

#include "entt/fwd.hpp"
#include "Properties.h"

namespace Bcg {
    struct GetPrimitives {
        explicit GetPrimitives(entt::entity entity_id) : entity_id(entity_id) {}

        PropertyContainer *vertices() const;

        PropertyContainer *halfedges() const;

        PropertyContainer *edges() const;

        PropertyContainer *faces() const;

        entt::entity entity_id;
    };
}

#endif //ENGINE24_GETPRIMITIVES_H
