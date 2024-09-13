//
// Created by alex on 13.09.24.
//

#ifndef ENGINE24_VIEWMATRIX_H
#define ENGINE24_VIEWMATRIX_H

#include "ViewParameters.h"
#include "RigidTransform.h"

namespace Bcg{

    class ViewMatrix : public RigidTransform {
    public:
        explicit ViewMatrix(const RigidTransform &model) : RigidTransform(model.inverse().matrix()) {

        }

        explicit ViewMatrix(const ViewParameters<float> &params) : ViewMatrix(params.eye(), params.center(), params.up()) {

        }

        explicit ViewMatrix(const Vector<float, 3> &eye, const Vector<float, 3> &center, const Vector<float, 3> &up) {
            RigidTransform t = RigidTransform::Identity();
            t.SetUp(up.normalized());
            t.SetDir((center - eye).normalized());
            t.SetRight(t.Up().cross(t.Dir()));
            t.SetPosition(-eye);
            m_matrix = t.matrix();
        }

        [[nodiscard]] RigidTransform model() const {
            return RigidTransform(m_matrix.inverse().eval());
        }
    };
}

#endif //ENGINE24_VIEWMATRIX_H
