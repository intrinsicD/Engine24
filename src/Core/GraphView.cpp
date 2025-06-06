//
// Created by alex on 06.06.25.
//

#include "GraphView.h"
#include "glad/gl.h"

namespace Bcg{
    void GraphView::draw() {
        glDrawElements(GL_LINES, num_indices, GL_UNSIGNED_INT, nullptr);
    }
}