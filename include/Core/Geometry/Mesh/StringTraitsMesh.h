//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MESHSTRINGTRAITS_H
#define ENGINE24_MESHSTRINGTRAITS_H

#include "StringTraits.h"
#include "SurfaceMesh.h"

namespace Bcg{
    template<>
    struct StringTraits<SurfaceMesh> {
        static std::string ToString(const SurfaceMesh &t) {
            return "Mesh to string not jet implemented";
        }
    };
}

#endif //ENGINE24_MESHSTRINGTRAITS_H
