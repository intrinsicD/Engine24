//
// Created by alex on 03.06.25.
//

#include "CommandsMeshView.h"
#include "ModuleMeshView.h"

namespace Bcg::Commands {
    void Commands::Setup<MeshView>::execute() const {
        ModuleMeshView::setup(entity_id);
    }

    void Commands::Cleanup<MeshView>::execute() const {
        ModuleMeshView::cleanup(entity_id);
    }

    void Commands::SetPositionMeshView::execute() const {
        ModuleMeshView::set_positions(entity_id, property_name);
    }

    void Commands::SetNormalMeshView::execute() const {
        ModuleMeshView::set_normals(entity_id, property_name);
    }

    void Commands::SetColorMeshView::execute() const {
        ModuleMeshView::set_colors(entity_id, property_name);
    }

    void Commands::SetUniforColorMeshView::execute() const {
        ModuleMeshView::set_uniform_color(entity_id, color);
    }

    void Commands::SetScalarfieldMeshView::execute() const {
        ModuleMeshView::set_scalarfield(entity_id, property_name);
    }

    void Commands::SetTrianglesMeshView::execute() const {
        ModuleMeshView::set_triangles(entity_id, tris);
    }
} // namespace Bcg::Commands