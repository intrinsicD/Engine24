//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_DRAWCALL_H
#define ENGINE24_DRAWCALL_H

namespace Bcg {
    struct DrawArrays {
        enum Mode {
            POINTS,
            LINE_STRIP,
            LINE_LOOP,
            LINES,
            LINE_STRIP_ADJACENCY,
            LINES_ADJACENCY,
            TRIANGLE_STRIP,
            TRIANGLE_FAN,
            TRIANGLES,
            TRIANGLE_STRIP_ADJACENCY,
            TRIANGLES_ADJACENCY,
            PATCHES
        };
        unsigned int mode;
        int first;
        unsigned int count;
    };

    struct DrawElements {
        enum Mode {
            POINTS,
            LINE_STRIP,
            LINE_LOOP,
            LINES,
            LINE_STRIP_ADJACENCY,
            LINES_ADJACENCY,
            TRIANGLE_STRIP,
            TRIANGLE_FAN,
            TRIANGLES,
            TRIANGLE_STRIP_ADJACENCY,
            TRIANGLES_ADJACENCY,
            PATCHES
        };
        unsigned int mode;
        unsigned int size;
        unsigned int type;
        const void *indices;
    };
}

#endif //ENGINE24_DRAWCALL_H
