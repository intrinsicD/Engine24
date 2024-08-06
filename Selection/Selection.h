//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_SELECTION_H
#define ENGINE24_SELECTION_H

#include <set>

namespace Bcg{
    struct Selection{
        bool show_vertices = false;
        std::set<unsigned int> vertices;
    };
}

#endif //ENGINE24_SELECTION_H
