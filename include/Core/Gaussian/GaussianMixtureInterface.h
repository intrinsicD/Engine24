#pragma once

#include <vector>

#include "PointCloudInterface.h"
#include "Logger.h"
#include "Eigen/Dense.h"

namespace Bcg {
    class GaussianMixtureInterface : public PointCloudInterface {
    public:
        VertexProperty<PointType> means;
        VertexProperty<Matrix<float, 3, 3> > covariances;
        VertexProperty<Matrix<float, 3, 3> > covariances_inv;
        VertexProperty<float> weights;

        explicit GaussianMixtureInterface(PointCloudData &data) : PointCloudInterface(data.vertices) {
        }

        explicit GaussianMixtureInterface(Vertices &vertices) : PointCloudInterface(vertices),
                                                                means(vertices.vertex_property<PointType>("v:point")),
                                                                covariances(
                                                                    vertices.vertex_property<Matrix<float, 3, 3> >(
                                                                        "v:covs")),
                                                                covariances_inv(
                                                                    vertices.vertex_property<Matrix<float, 3, 3> >(
                                                                        "v:covs_inv")),
                                                                weights(vertices.vertex_property<float>(
                                                                    "v:weights")) {
            assert(means && means.name() == "v:point");
            assert(covariances && covariances.name() == "v:covs");
            assert(covariances_inv && covariances_inv.name() == "v:covs_inv");
            assert(weights && weights.name() == "v:weights");
        }

        GaussianMixtureInterface() = default;


        void set_covs(const std::vector<Vector<float, 3> > &scaling,
                      const std::vector<Vector<float, 4> > &quaternions) {
            if (scaling.size() != quaternions.size()) {
                Log::Error("Size of scaling does not match Size of quaternions");
                return;
            }
            if (scaling.size() != vertices.size()) {
                Log::Error("Size of scaling does not match Size of vertices");
                return;
            }
            for (size_t i = 0; i < scaling.size(); ++i) {
                glm::quat q = {quaternions[i].w, quaternions[i].x, quaternions[i].y, quaternions[i].z};

                // Convert quaternion to rotation matrix using GLM
                Matrix<float, 3, 3> rotation_matrix = glm::mat3_cast(q);

                // Create scale matrix
                Matrix<float, 3, 3> scale_matrix = Matrix<float, 3, 3>(0.0f);
                scale_matrix[0][0] = scaling[i].x * scaling[i].x;
                scale_matrix[1][1] = scaling[i].y * scaling[i].y;
                scale_matrix[2][2] = scaling[i].z * scaling[i].z;

                // Compute covariance matrix: R * S^2 * R^T
                covariances[Vertex(i)] = rotation_matrix * scale_matrix * glm::transpose(rotation_matrix);
                // Precompute inverse covariance matrix
                covariances_inv[Vertex(i)] = glm::inverse(covariances[Vertex(i)]);
            }
        }

        std::vector<float> pdf(const std::vector<Vector<float, 3> > &query_points) const;

        std::vector<Vector<float, 3> > gradient(const std::vector<Vector<float, 3> > &query_points) const;

        std::vector<Vector<float, 3> > normal(const std::vector<Vector<float, 3> > &query_points) const;

        std::vector<Matrix<float, 3, 3> > hessian(const std::vector<Vector<float, 3> > &query_points) const;
    };
}
