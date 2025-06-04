//
// Created by alex on 04.06.25.
//

#include "PhongSplattingView.h"
#include "glad/gl.h"

namespace Bcg{
    void PhongSplattingView::draw() {
        glDrawElements(GL_POINTS, num_points, GL_UNSIGNED_INT, nullptr);
    }
}