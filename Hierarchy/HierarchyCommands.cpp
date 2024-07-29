//
// Created by alex on 29.07.24.
//

#include "HierarchyCommands.h"
#include "PluginHierarchy.h"

namespace Bcg {
    void UpdateTransformsDeferred::execute() const {
        PluginHierarchy::update_transforms(entity_id);
    }
}