//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_IRENDERPASS_H
#define ENGINE24_IRENDERPASS_H

#include "entt/fwd.hpp"

namespace Bcg{
    struct Camera;

    class IRenderPass{
    public:
        virtual ~IRenderPass() = default;
        // Each pass executes its logic using the provided scene data.
        virtual void execute(entt::registry& registry, const Camera& camera) = 0;
    };
}
#endif //ENGINE24_IRENDERPASS_H
