//
// Created by alex on 6/1/25.
//

#include "CommandsAABB.h"
#include "ModuleAABB.h"

namespace Bcg::Commands {
    void Commands::Setup<AABB>::execute() const {
        ModuleAABB::setup(entity_id);
    }

    void Commands::Cleanup<AABB>::execute() const {
        ModuleAABB::cleanup(entity_id);
    }

    void Commands::CenterAndScaleByAABB::execute() const {
        ModuleAABB::center_and_scale_by_aabb(entity_id, property_name);
    }
}
