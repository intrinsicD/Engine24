//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_GAUSSIANMIXTURE_H
#define ENGINE24_GAUSSIANMIXTURE_H

#include "Properties.h"
#include "MatVec.h"
#include "Rotation.h"


namespace Bcg {
    template<typename T>
    class GaussianMixture {
    public:
        Property<Vector<T, 3>> means;
        Property<Vector<T, 3>> scales;
        Property<AngleAxis<T>> rotations;
        Property<T> weights;

        PropertyContainer gaussians;

        GaussianMixture() : means(gaussians.add("means", Vector<T, 3>(0.0))),
                            scales(gaussians.add("scales", Vector<T, 3>(1.0))),
                            rotations(gaussians.add("rotations", AngleAxis<T>())),
                            weights(gaussians.add("weights", T(1.0))) {

        }

        Matrix<T, 3, 3> get_cov(int i) {
            Matrix<T, 3, 3> rot = Matrix<T, 3, 3>(rotations[i].matrix());
            Matrix<T, 3, 3> scale = Matrix<T, 3, 3>::Identity();
            for (int j = 0; j < 3; ++j) {
                scale[j][j] = scales[i][j] * scales[i][j];
            }
            return glm::transpose(rot) * scale * rot;
        }

        Matrix<T, 3, 3> get_cov_inv(int i) {
            Matrix<T, 3, 3> rot = Matrix<T, 3, 3>(rotations[i].matrix());
            Matrix<T, 3, 3> scale = Matrix<T, 3, 3>::Identity();
            for (int j = 0; j < 3; ++j) {
                scale[j][j] = 1 / scales[i][j] * scales[i][j];
            }
            return glm::transpose(rot) * scale * rot;
        }

        T pdf(const Vector<T, 3> &point) {
            T result = 0;
            for (size_t i = 0; i < means.size(); ++i) {
                Vector<T, 3> diff = point - means[i];
                result += weights[i] * std::exp(-dot(diff, get_cov_inv(i) * diff) / 2.0);
            }
            return result;
        }

        Vector<T, 3> grad(const Vector<T, 3> &point) {
            Vector<T, 3> result = Vector<T, 3>::Zero();
            for (size_t i = 0; i < means.size(); ++i) {
                Vector<T, 3> diff = point - means[i];
                Vector<T, 3> diff2 = get_cov_inv(i) * (point - means[i]);
                result += weights[i] * std::exp(-dot(diff, diff2) / 2.0) * diff2;
            }

            return result;
        }

        Matrix<T, 3, 3> hess(const Vector<T, 3> &point) {
            Matrix<T, 3, 3> result = Matrix<T, 3, 3>::Zero();
            for (size_t i = 0; i < means.size(); ++i) {
                Vector<T, 3> diff = point - means[i];
                Matrix<T, 3, 3> cov_inv = get_cov_inv(i);
                Vector<T, 3> diff2 = cov_inv * (point - means[i]);
                result += weights[i] * std::exp(-dot(diff, diff2) / 2.0) * (glm::outerProduct(diff2, diff2) - cov_inv);
            }

            return result;
        }

        void set_cov(int i, const Matrix<T, 3, 3> &cov) {
            //decompose this covariance matrix into rotation and scale
            //then set rot and scale
        }

        void new_gaussian() {
            gaussians.push_back();
        }

        void new_gaussian(const Vector<T, 3> &mean, const Matrix<T, 3, 3> &covariance, T weight = 1.0) {
            gaussians.push_back();
            size_t i = gaussians.size() - 1;
            means[i] = mean;
            set_cov(i, covariance);
            weights[i] = weight;
        }


    };
}

#endif //ENGINE24_GAUSSIANMIXTURE_H
