//
// Created by alex on 12.08.24.
//

#ifndef ENGINE24_FWD_H
#define ENGINE24_FWD_H

namespace Bcg{
    class SurfaceMesh;

    template<typename T>
    class AABBbase;

    using AABBf = class AABBbase<float>;
    using AABB = AABBf;
}

#endif //ENGINE24_FWD_H
