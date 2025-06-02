//
// Created by alex on 02.06.25.
//

#include "CommandsTransform.h"
#include "ModuleTransform.h"

namespace Bcg::Commands {
    void Setup<Transform>::execute() const {
        ModuleTransform::setup(entity_id);
    }

    void Cleanup<Transform>::execute() const {
        ModuleTransform::cleanup(entity_id);
    }

    void SetIdentityTransform::execute() const {
        ModuleTransform::set_identity_transform(entity_id);
    }

} // namespace Bcg::Commands