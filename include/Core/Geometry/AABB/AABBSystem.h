#pragma once

#include <entt/fwd.hpp>

namespace Bcg {
    void UpdateWorldAABBSystem(entt::registry &registry);

    void ClearWorldAABBDirtyTags(entt::registry &registry);
}