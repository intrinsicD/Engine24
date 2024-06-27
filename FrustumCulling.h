//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_FRUSTUMCULLING_H
#define ENGINE24_FRUSTUMCULLING_H

#include "Plugin.h"
#include "entt/fwd.hpp"

namespace Bcg {
    struct Frustum;
    struct Sphere;
    struct AABB;
    struct OBB;
    struct ConvexHull;

    class FrustumCulling : public Plugin {
    public:
        explicit FrustumCulling();

        ~FrustumCulling() override = default;

        static bool is_visible(const Frustum &frustum, entt::entity entity_id);

        static bool is_visible(const Frustum &frustum, const Sphere &sphere);

        static bool is_visible(const Frustum &frustum, const AABB &aabb);

        static bool is_visible(const Frustum &frustum, const OBB &obb);

        static bool is_visible(const Frustum &frustum, const ConvexHull &hull);

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;

    protected:
        const char *name;
    };
}

#endif //ENGINE24_FRUSTUMCULLING_H
