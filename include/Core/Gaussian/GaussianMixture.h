//
// Created by alex on 28.05.25.
//

#ifndef ENGINE24_GAUSSIANMIXTURE_H
#define ENGINE24_GAUSSIANMIXTURE_H

#include "PointCloud.h"

namespace Bcg {
    class GaussianMixture : public PointCloud {
    public:
        VertexProperty<Vector<float, 3>> means;
        VertexProperty<Matrix<float, 3, 3>> covariances;
        VertexProperty<float> weights;

        GaussianMixture() : PointCloud() {
            means = add_vertex_property<Vector<float, 3>>("means");
            covariances = add_vertex_property<Matrix<float, 3, 3>>("covariances");
            weights = add_vertex_property<float>("weights");
        }

        float pdf(const Vector<float, 3> &x) const;

        Vector<float, 3> gradient(const Vector<float, 3> &x) const;

        Matrix<float, 3, 3> hessian(const Vector<float, 3> &x) const;

        Vector<float, 3> normal(const Vector<float, 3> &x) const{
            Vector<float, 3> grad = gradient(x);
            float norm = length(grad);
            if (norm > 0) {
                return grad / norm;
            } else {
                return Vector<float, 3>(0.0f);
            }
        }

        Matrix<float, 3, 3> ortho_projector(const Vector<float, 3> &x) const {
            Vector<float, 3> n = normal(x);
            return Matrix<float, 3, 3>(1.0f) - outerProduct(n, n);
        }

        Matrix<float, 3, 3> second_fundamental_form(const Vector<float, 3> &x) const {
            Vector<float, 3> grad = gradient(x);
            Matrix<float, 3, 3> hess = hessian(x);
            Matrix<float, 3, 3> p = ortho_projector(x);
            return p * hess * p / length(grad);
        }

        Vertex new_gaussian() {
            return new_vertex();
        }

        Vertex add_gaussian(const Vector<float, 3> &mean, const Matrix<float, 3, 3> &covariance, float weight = 1.0f) {
            Vertex v = new_gaussian();
            means[v] = mean;
            covariances[v] = covariance;
            weights[v] = weight;
            return v;
        }
    };
}

#endif //ENGINE24_GAUSSIANMIXTURE_H
