#pragma once

#include <vector>

#include "GeometryCommon.h"
#include "Eigen/Dense.h"

namespace Bcg {
    template<typename T>
    class GaussianMixtureInterface {
    public:
        using ScalarType = T;
        using VecType = Eigen::Vector<ScalarType, 3>;
        using MatType = Eigen::Matrix<ScalarType, 3, 3>;

        Vertices &vertices;
        VertexProperty<VecType> means;
        VertexProperty<MatType> covariances;
        VertexProperty<MatType> covariances_inv;
        VertexProperty<ScalarType> weights;

        explicit GaussianMixtureInterface(Vertices &vertices) : vertices(vertices) {
            means = vertices.vertex_property<VecType>("means");
            covariances = vertices.vertex_property<MatType>("covariances");
            weights = vertices.vertex_property<ScalarType>("weights");
        }

        GaussianMixtureInterface() = default;

        void set_means(const std::vector<VecType> &means_data) {
            means = vertices.vertex_property<VecType>>("means");
            means.vector() = means_data;
        }

        void set_covs(const std::vector<MatType> &covs_data) {
            covariances = vertices.vertex_property<MatType>("covariances");
            covariances.vector() = covs_data;
        }

        void set_covs(const std::vector<VecType> &scaling, const std::vector<VecType> &quaternions) {
            covariances = vertices.vertex_property<MatType>("covariances");
            covariances.vector() = covs_data;
        }

        void set_weights(const std::vector<ScalarType> &weights_data) {
            weights = vertices.vertex_property<float>("weights");
            weights.vector() = weights_data;
        }

        void compute_inverse_covariances() const;

        std::vector<float> pdf(const std::vector<VecType> &query_points) const;
        std::vector<VecType> gradient(const std::vector<VecType> &query_points) const;
        std::vector<VecType> normal(const std::vector<VecType> &query_points) const;
        std::vector<MatType> hessian(const std::vector<VecType> &query_points) const;
    };
}
