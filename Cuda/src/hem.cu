//
// Created by alex on 18.08.24.
//

#include "hem.cuh"
#include "Cuda/Hem.h"

namespace Bcg::cuda {
    HemResult Hem(const std::vector<Vector<float, 3>> &positions, unsigned int levels) {
        std::vector<vec3> ps(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i].x(), positions[i].y(), positions[i].z()};
        }

        hem mixture;
        hem_params params;
        params.nLevels = levels;
        mixture = hem(ps, params);
        mixture.fit();

        auto h_means = mixture.means_host();
        auto h_covs = mixture.covs_host();
        auto h_weights = mixture.weights_host();
        auto h_nvars = mixture.normal_variance_host();

        HemResult result;
        result.means.resize(h_means.size());
        result.covs.resize(h_covs.size());
        result.weights.resize(h_weights.size());
        result.nvars.resize(h_nvars.size());

        for (size_t i = 0; i < h_means.size(); ++i) {
            result.means[i] = {h_means[i].x, h_means[i].y, h_means[i].z};
            result.covs[i] << h_covs[i].col0.x, h_covs[i].col0.y, h_covs[i].col0.z,
                    h_covs[i].col1.x, h_covs[i].col1.y, h_covs[i].col1.z,
                    h_covs[i].col2.x, h_covs[i].col2.y, h_covs[i].col2.z;
            result.weights[i] = h_weights[i];
            result.nvars[i] = {h_nvars[i].x, h_nvars[i].y, h_nvars[i].z};
        }
        return result;
    }
}