//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_FRUSTUM_H
#define ENGINE24_FRUSTUM_H

#include "Plane.h"

namespace Bcg {
    template<typename T>
    struct FrustumBase {
        union {
            Matrix<T, 6, 4> matrix;
            PlaneBase<T> array[6];
        } planes;
        union {
            Matrix<T, 8, 3> matrix;
            Vector<T, 3> array[8];
            std::vector<Vector<T, 3>> vector;
        } vertices;

        std::vector<Vector<T, 3>> Edges() const {
            return {vertices.array[1] - vertices.array[0],
                    vertices.array[2] - vertices.array[1],
                    vertices.array[3] - vertices.array[2],
                    vertices.array[0] - vertices.array[3],
                    vertices.array[5] - vertices.array[4],
                    vertices.array[6] - vertices.array[5],
                    vertices.array[7] - vertices.array[6],
                    vertices.array[4] - vertices.array[7]};
        }
    };

    using Frustumf = FrustumBase<float>;
    using Frustum = Frustumf;
}

#endif //ENGINE24_FRUSTUM_H
