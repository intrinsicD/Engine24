//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_ROTATION_H
#define ENGINE24_ROTATION_H

#include "MatVec.h"

namespace Bcg::Rotation {
    using Quaternion = Eigen::Quaternion<float>;

    Matrix<float, 3, 3> CrossProductMatrix(const Vector<float, 3> &vector) {
        Matrix<float, 3, 3> matrix;
        matrix(1, 0) = vector[2];
        matrix(0, 1) = -vector[2];
        matrix(0, 2) = vector[1];
        matrix(2, 0) = -vector[1];
        matrix(2, 1) = vector[0];
        matrix(1, 2) = -vector[0];
        return matrix;
    }

    struct AngleAxis : public Vector<float, 3> {

    };

    struct Cayley : public Vector<float, 3> {

    };

    struct TwoAxis : public Matrix<float, 3, 2> {

    };
}

#endif //ENGINE24_ROTATION_H
