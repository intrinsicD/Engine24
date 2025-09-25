//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_TRANSFORMSYSTEM_H
#define ENGINE24_TRANSFORMSYSTEM_H

#include "entt/fwd.hpp"

namespace Bcg{
    void UpdateTransformSystem(entt::registry &registry);
    void ClearTransformDirtyTags(entt::registry &registry);
}

#endif //ENGINE24_TRANSFORMSYSTEM_H
