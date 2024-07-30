//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_POINTCLOUDCOMPUTE_H
#define ENGINE24_POINTCLOUDCOMPUTE_H

#include "PointCloud.h"
#include "entt/fwd.hpp"

namespace Bcg {
    Property<Normal> ComputeVertexNormals(entt::entity entity_id, PropertyContainer &vertices, int k);
}
#endif //ENGINE24_POINTCLOUDCOMPUTE_H
