//
// Created by alex on 12.08.24.
//

#ifndef ENGINE24_ENTITY_H
#define ENGINE24_ENTITY_H

#include "fmt/core.h"
#include "fmt/format.h"
#include "entt/fwd.hpp"


namespace fmt {
    template<>
    struct formatter<entt::entity> : formatter<uint32_t> {
        template<typename FormatContext>
        auto format(entt::entity entity_id, FormatContext &ctx) const {
            return formatter<uint32_t>::format(static_cast<uint32_t>(entity_id), ctx);
        }
    };
}

namespace Bcg {
    class Entity {
    public:
        Entity();

        explicit Entity(entt::entity entity_id);

        operator bool() const;

        bool is_valid() const;

        operator entt::entity() const;

        Entity &create();

        Entity &destroy();

        [[nodiscard]] entt::entity id() const;

        [[nodiscard]] std::string to_string() const;

    private:
        entt::entity entity_id;
    };
}

namespace fmt {
    template<>
    struct formatter<Bcg::Entity> : formatter<uint32_t> {
        template<typename FormatContext>
        auto format(Bcg::Entity entity, FormatContext &ctx) const {
            return formatter<uint32_t>::format(static_cast<uint32_t>(entity.id()), ctx);
        }
    };
}
#endif //ENGINE24_ENTITY_H
