//
// Created by alex on 13.09.24.
//

#ifndef ENGINE24_AABBUNION_H
#define ENGINE24_AABBUNION_H

#include "AABBStruct.h"

namespace Bcg {
    template<typename T, int N>
    inline AABBStruct<T, N> Union(const AABBStruct<T, N> &a, const AABBStruct<T, N> &b) {
        AABBStruct<T, N> result = a;
        result.grow(b.min);
        result.grow(b.max);
        return result;
    }
}

#endif //ENGINE24_AABBUNION_H
