//
// Created by alex on 27.06.24.
//

#include "FrustumCulling.h"
#include "entt/entt.hpp"

namespace Bcg {
    struct Frustum;
    struct Sphere;
    struct AABB;
    struct OBB;
    struct ConvexHull;


    FrustumCulling::FrustumCulling() : Plugin("FrustumCulling") {

    }

    bool FrustumCulling::is_visible(const Frustum &frustum, entt::entity entity_id) {

    }

    bool FrustumCulling::is_visible(const Frustum &frustum, const Sphere &sphere) {

    }

    bool FrustumCulling::is_visible(const Frustum &frustum, const AABB &aabb) {

    }

    bool FrustumCulling::is_visible(const Frustum &frustum, const OBB &obb) {

    }

    bool FrustumCulling::is_visible(const Frustum &frustum, const ConvexHull &hull) {

    }

    void FrustumCulling::activate() {}

    void FrustumCulling::begin_frame() {}

    void FrustumCulling::update() {}

    void FrustumCulling::end_frame() {}

    void FrustumCulling::deactivate() {}

    void FrustumCulling::render_menu() {}

    void FrustumCulling::render_gui() {
    }

    void FrustumCulling::render() {}

}