//
// Created by alex on 04.06.25.
//

#include "CommandsSphereView.h"
#include "ModuleSphereView.h"

namespace Bcg::Commands {
    void Commands::Setup<SphereView>::execute() const {
        ModuleSphereView::setup(entity_id);
    }

    void Commands::SetPositionSphereView::execute() const {
        ModuleSphereView::set_position(entity_id, property_name);
    }

    void Commands::SetRadiusSphereView::execute() const {
        ModuleSphereView::set_radius(entity_id, property_name);
    }

    void Commands::SetUniformRadiusSphereView::execute() const {
        ModuleSphereView::set_uniform_radius(entity_id, uniform_radius);
    }

    void Commands::SetColorSphereView::execute() const {
        ModuleSphereView::set_color(entity_id, property_name);
    }

    void Commands::SetUniformColorSphereView::execute() const {
        ModuleSphereView::set_uniform_color(entity_id, uniform_color);
    }

    void Commands::SetScalarfieldSphereView::execute() const {
        ModuleSphereView::set_scalarfield(entity_id, property_name);
    }

    void Commands::SetNormalSphereView::execute() const {
        ModuleSphereView::set_normal(entity_id, property_name);
    }

    void Commands::SetIndicesSphereView::execute() const {
        ModuleSphereView::set_indices(entity_id, indices);
    }
}