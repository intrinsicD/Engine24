//
// Created by alex on 04.08.24.
//

#include "MeshView.h"
#include "glad/gl.h"

namespace Bcg{
    void MeshView::draw() {
        vao.bind();
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
    }
}