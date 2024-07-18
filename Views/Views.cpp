//
// Created by alex on 18.07.24.
//
#include "Views.h"
#include "glad/gl.h"

namespace Bcg {
    void TriangleView::draw() {
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
    }
}