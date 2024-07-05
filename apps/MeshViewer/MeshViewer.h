//
// Created by alex on 05.07.24.
//

#ifndef ENGINE24_MESHVIEWER_H
#define ENGINE24_MESHVIEWER_H

#include "SurfaceMesh.h"

namespace Bcg {
    class MeshViewer {
    public:
        MeshViewer();

        void run();

        unsigned int vao, vbo, ebo;
        unsigned int program;
    };
}

#endif //ENGINE24_MESHVIEWER_H
