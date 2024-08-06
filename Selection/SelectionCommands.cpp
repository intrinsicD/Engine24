//
// Created by alex on 06.08.24.
//

#include "SelectionCommands.h"
#include "SphereViewCommands.h"
#include "SphereView.h"
#include "Selection.h"
#include "Engine.h"
#include "GetPrimitives.h"
#include "PropertyEigenMap.h"

namespace Bcg::Commands {
    void MarkPoints::execute() const {
        if (!Engine::has<Selection>(entity_id)) {
            return;
        }
        auto *vertices = GetPrimitives(entity_id).vertices();
        auto selected_vertices = vertices->get_or_add<Vector<float, 3>>(property_name, Vector<float, 3>::Ones());

        Map(selected_vertices.vector()).setOnes();
        auto &selection = Engine::State().get<Selection>(entity_id);
        for (auto idx: selection.vertices) {
            selected_vertices[idx] = {1.0, 0.0, 0.0};
        }

        View::SetColorSphereView(entity_id, property_name).execute();
    }

    void EnableVertexSelection::execute() const {

    }
}