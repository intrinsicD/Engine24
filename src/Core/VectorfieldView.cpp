//
// Created by alex on 05.08.24.
//

#include "VectorfieldView.h"
#include "glad/gl.h"

namespace Bcg{
    void VectorfieldView::draw() {
        glDrawElements(GL_POINTS, num_vectors, GL_UNSIGNED_INT, nullptr);
    }
}
