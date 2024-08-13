//
// Created by alex on 02.08.24.
//

#include "SphereView.h"
#include "glad/gl.h"

namespace Bcg{
    void SphereView::draw() {
        glDrawElements(GL_POINTS, num_spheres, GL_UNSIGNED_INT, nullptr);
    }
}