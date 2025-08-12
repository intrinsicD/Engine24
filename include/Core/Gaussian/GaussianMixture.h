//
// Created by alex on 28.05.25.
//

#ifndef ENGINE24_GAUSSIANMIXTURE_H
#define ENGINE24_GAUSSIANMIXTURE_H

#include "PointCloud.h"

namespace Bcg {
    class GaussianMixture {
    public:
        Vertices vprops_;
        VertexProperty<Vector<float, 3>> means;
        VertexProperty<Matrix<float, 3, 3>> covariances;
        VertexProperty<float> weights;

        explicit GaussianMixture(PointCloud &pc) {
            vprops_ = pc.vprops_;
        }

        GaussianMixture() = default;

        void set_means(const std::vector<PointType> &means_data) {
            means = vprops_.vertex_property<Vector<float, 3>>("means");
            means.vector() = means_data;
        }

        void set_covs(const std::vector<Matrix<float, 3, 3>> &covs_data) {
            covariances = vprops_.vertex_property<Matrix<float, 3, 3>>("covariances");
            covariances.vector() = covs_data;
        }

        void set_weights(const std::vector<float> &weights_data) {
            weights = vprops_.vertex_property<float>("weights");
            weights.vector() = weights_data;
        }

        void fit(const std::vector<PointType> &means_data, size_t num_gaussians);

        float pdf(const Vector<float, 3> &x) const;

        Vector<float, 3> gradient(const Vector<float, 3> &x) const;

        Matrix<float, 3, 3> hessian(const Vector<float, 3> &x) const;

        Vector<float, 3> normal(const Vector<float, 3> &x) const;

        Matrix<float, 3, 3> ortho_projector(const Vector<float, 3> &x) const;

        Matrix<float, 3, 3> second_fundamental_form(const Vector<float, 3> &x) const;
    };
}

#endif //ENGINE24_GAUSSIANMIXTURE_H
