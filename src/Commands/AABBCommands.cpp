//
// Created by alex on 6/1/25.
//

#include "CommandsAABB.h"
#include "ModuleAABB.h"

namespace Bcg::Commands {
    void Commands::Setup<AABB<float>>::execute() const {
        ModuleAABB::setup(entity_id);
    }

    void Commands::Cleanup<AABB<float>>::execute() const {
        ModuleAABB::cleanup(entity_id);
    }
}
