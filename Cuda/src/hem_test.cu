//
// Created by alex on 18.08.24.
//

#include "Cuda/Hem.h"
#include "lbvh.cuh"
#include "mat3.cuh"
#include "vec_operations.cuh"
#include "gaussian.cuh"
#include "Logger.h"
#include <float.h>
#include <thrust/random.h>
#include <thrust/gather.h>

namespace Bcg::cuda {
    using hbvh = lbvh<vec3, aabb_getter>;
    using dbvh = bvh_device<vec3>;

    struct HemDeviceDataPtr {
        vec3 *means;
        mat3 *covs;
        float *weights;
        vec3 *nvars;
        bool *is_parent;
        unsigned int *parents_indices;
        unsigned int *children_indices;

        dbvh d_bvh_means;

        int num_components;
        int num_parents;
        int num_children;
    };

    struct random_key_generator {
        unsigned int seed;

        __host__ __device__
        random_key_generator(unsigned int seed_val) : seed(seed_val) {}

        __host__ __device__
        float operator()(unsigned int i) const {
            thrust::default_random_engine rng(seed);
            thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
            rng.discard(i);  // Ensure each index gets a different random value
            return dist(rng);
        }
    };

    struct distance_calculator {
        __device__ __host__
        float operator()(const vec3 &point, const vec3 &object) const noexcept {
            return (point.x - object.x) * (point.x - object.x) +
                   (point.y - object.y) * (point.y - object.y) +
                   (point.z - object.z) * (point.z - object.z);
        }
    };

    __device__ float atomicMaxFloat(float *address, float val) {
        int *address_as_int = (int *) address;
        int old = *address_as_int, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);

        return __int_as_float(old);
    }


    __device__ __host__ float max_eigenvalues_radius(const mat3 &cov) {
        mat3 evecs;
        return jacobi_eigen(cov, evecs).z;
    }

    __device__ __host__ float geroshin_radius(const mat3 &cov) {
        return cov.geroshin_radius();
    }

    __device__ __host__ float trace_radius(const mat3 &cov) {
        return cov.trace() / 3.0f;
    }

    __device__ __host__ float laplacian_bound(const mat3 &cov) {
        return cov.norm();
    }

    __device__ __host__ float max_row_sum(const mat3 &cov) {
        return cov.max_row_sum();
    }

    __device__ __host__ float hem_likelihood(const vec3 &mean_parent, const mat3 &inv_cov_parent,
                                             const vec3 &mean_child, const mat3 &cov_child, float child_weight) {
        vec3 diff = mean_parent - mean_child;
        float smd = diff.dot(inv_cov_parent * diff);
        mat3 ipcCov = inv_cov_parent * cov_child;
        float ipcTrace = ipcCov.trace();
        float e = -0.5f * (smd + ipcTrace);
        float f = 0.063493635934241f * sqrtf(inv_cov_parent.determinant()) * expf(e);
        return powf(f, child_weight);
    }

    struct HemDeviceData {
        thrust::device_vector<vec3> means;
        thrust::device_vector<mat3> covs;
        thrust::device_vector<float> weights;
        thrust::device_vector<vec3> nvars;
        thrust::device_vector<bool> is_parent;
        thrust::device_vector<unsigned int> parents_indices;
        thrust::device_vector<unsigned int> children_indices;
        hbvh h_bvh;

        HemDeviceDataPtr get_device_repr() {
            HemDeviceDataPtr d_hem;
            d_hem.means = means.data().get();
            d_hem.covs = covs.data().get();
            d_hem.weights = weights.data().get();
            d_hem.nvars = nvars.data().get();
            d_hem.is_parent = is_parent.data().get();
            d_hem.parents_indices = parents_indices.data().get();
            d_hem.children_indices = children_indices.data().get();
            d_hem.num_components = means.size();
            d_hem.num_parents = parents_indices.size();
            d_hem.num_children = children_indices.size();
            d_hem.d_bvh_means = h_bvh.get_device_repr();
            return d_hem;
        }
    };

    struct HemParams {
        float globalInitRadius = 1.0f;        // global initialization kernel radius (applies only if useGlobalInitRadius == true)
        bool useGlobalInitRadius = true;        // use global initialization radius instead of NNDist sampling (more stable for non-uniform point sampling)
        uint nNNDistSamples = 10;            // number of sample points for automatic initialization radius estimation (more is better)
        bool useWeightedPotentials = true;    // if true, performs WLOP-like balancing of the initial Gaussian potentials
        float alpha0 = 2.5f;                    // multiple of nearest neighbor distance to use for initial query radius (<= 2.5f recommended)
        float alpha = 2.5f;                    // multiple of cluster maximum std deviation to use for query radius (<= 2.5f recommended)
        uint nLevels = 5;                    // number of levels to use for hierarchical clustering
        float hemReductionFactor = 3.0f;        // factor by which to reduce the mixture each level (3 is recommended, don't change!)
        uint knn = 10;                        // number of nearest neighbors to use for kNN initialization
    };

    void SeparateParentsChildren(HemDeviceData &data) {
        //----------------------------------------------------------------------------------------------------------
        // Separate components into parents and children randomly (parents are the first num_parents components)
        //----------------------------------------------------------------------------------------------------------
        unsigned int num_components = data.means.size();
        unsigned int num_parents = num_components / 3;

        thrust::device_vector<unsigned int> component_indices(num_components);
        thrust::sequence(component_indices.begin(), component_indices.end());

        // Step 2: Generate random keys for shuffling
        thrust::device_vector<float> random_keys(num_components);
        unsigned int seed = 1234; // Or any arbitrary seed
        thrust::transform(thrust::device, component_indices.begin(), component_indices.end(), random_keys.begin(),
                          random_key_generator(seed));

        // Step 3: Sort indices by the random keys to shuffle them
        thrust::sort_by_key(random_keys.begin(), random_keys.end(), component_indices.begin());

        // Step 4: Split into parents and children
        thrust::device_vector<unsigned int> parents_indices(component_indices.begin(),
                                                            component_indices.begin() + num_parents);
        thrust::device_vector<unsigned int> children_indices(component_indices.begin() + num_parents,
                                                             component_indices.end());

        data.parents_indices = std::move(parents_indices);
        data.children_indices = std::move(children_indices);

        bool *d_is_parent = data.is_parent.data().get();

        thrust::for_each(thrust::device,
                         data.parents_indices.begin(), data.parents_indices.end(),
        [d_is_parent]
                __device__(unsigned int
        parent_idx) {
            d_is_parent[parent_idx] = true;
        });
    }

    HemDeviceData InitMixture(const std::vector<vec3> &positions, HemParams params) {
        HemDeviceData parents;
        parents.means = positions;
        parents.covs.resize(positions.size());
        parents.weights.resize(positions.size());
        parents.nvars.resize(positions.size());
        parents.is_parent.resize(positions.size());

        SeparateParentsChildren(parents);

        parents.h_bvh = hbvh(positions.begin(), positions.end(), false);

        HemDeviceDataPtr d_parents = parents.get_device_repr();

        float max_radius = 0.0f;
        float *d_max_radius = &max_radius;
        // Allocate memory on the device
        cudaMalloc(&d_max_radius, sizeof(float));

// Initialize the value in device memory
        cudaMemcpy(d_max_radius, &max_radius, sizeof(float), cudaMemcpyHostToDevice);
        //knn query to initialize the search radius
        //TODO check if this is sufficient or if we have to do it like in the original implementation
        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(parents.parents_indices.size()),
                         [d_parents, d_max_radius, params] __device__(unsigned int idx) {
                             unsigned int parent_idx = d_parents.parents_indices[idx];
                             unsigned int indices[32];
                             const vec3 &query_point = d_parents.means[parent_idx];
                             auto nn = query_device(d_parents.d_bvh_means, knn(query_point, params.knn),
                                                    distance_calculator(),
                                                    indices, 32);
                             for (int i = 0; i < nn; ++i) {
                                 unsigned int neighbor_idx = indices[i];
                                 vec3 neighbor = d_parents.means[neighbor_idx];
                                 float radius = (neighbor - query_point).length();
                                 atomicMaxFloat(d_max_radius, radius);
                             }
                         });

        cudaMemcpy(&max_radius, d_max_radius, sizeof(float), cudaMemcpyDeviceToHost);
        Log::Info("Max radius: {}", max_radius * params.alpha0);

        float *d_alpha0 = &params.alpha0;
        cudaMalloc(&d_alpha0, sizeof(float));
        cudaMemcpy(d_alpha0, &params.alpha0, sizeof(float), cudaMemcpyHostToDevice);
        bool *d_useWeightedPotentials = &params.useWeightedPotentials;
        cudaMalloc(&d_useWeightedPotentials, sizeof(bool));
        cudaMemcpy(d_useWeightedPotentials, &params.useWeightedPotentials, sizeof(bool), cudaMemcpyHostToDevice);

        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(d_parents.num_components),
                         [d_parents, d_max_radius, d_alpha0, d_useWeightedPotentials] __device__(unsigned int idx) {
                             const vec3 &query_point = d_parents.means[idx];
                             float r = *d_max_radius * *d_alpha0;

                             unsigned int indices[64];
                             auto nn = query_device(d_parents.d_bvh_means, overlaps_sphere(query_point, r),
                                                    indices, 64);
                             const float minus_16_over_h2 = -16.0f / (r * r);

                             float eps = r * r * 0.0001f;
                             float density = 0.000001f;

                             vec3 mean = vec3::constant(0);
                             mat3 cov = mat3::identity() * eps;
                             nn = fminf(nn, 64);
                             for (int i = 0; i < nn; ++i) {
                                 unsigned int neighbor_idx = indices[i];
                                 vec3 neighbor = d_parents.means[neighbor_idx];
                                 vec3 diff = neighbor - query_point;
                                 cov = cov + outer(diff, diff);
                                 mean = mean + neighbor;
                                 density += expf(minus_16_over_h2 * diff.dot(diff));
                             }
                             float inv_w = 1.0f / fmaxf(nn, 1.0f);
                             vec3 o = mean * inv_w - query_point;
                             cov = cov * inv_w - outer(o, o);

                             d_parents.means[idx] = query_point;
                             d_parents.covs[idx] = conditionCov(cov);
                             d_parents.weights[idx] = *d_useWeightedPotentials ? 1.0f / density : 1.0f;
                             mat3 evecs;
                             jacobi_eigen(cov, evecs);

                             float initialVar = 0.001f;
                             d_parents.nvars[idx] = evecs.col0 * initialVar;
                         });

        cudaFree(d_max_radius);
        return parents;
    }

    template<typename T, int N>
    struct CudaMatrixRow {
        T data[N];
    };

    HemDeviceData ClusterLevel(HemDeviceData &components, HemParams params) {
        unsigned int num_components = components.means.size();
        unsigned int num_parents = components.parents_indices.size();
        HemDeviceData parents;
        parents.means.resize(num_parents);
        parents.covs.resize(num_parents);
        parents.nvars.resize(num_parents);
        parents.weights.resize(num_parents);
        parents.is_parent.resize(num_parents);


        auto d_parents = parents.get_device_repr();
        auto d_components = components.get_device_repr();

        thrust::device_vector<CudaMatrixRow<unsigned int, 64>> parents_children_indices(num_parents);
        thrust::device_vector<CudaMatrixRow<float, 64>> comp_likelihoods(num_components);
        thrust::device_vector<float> comp_likelihood_sums(num_components, 0);
        thrust::device_vector<float> parents_radii(num_parents);

        CudaMatrixRow<unsigned int, 64> *d_parents_children_indices = parents_children_indices.data().get();
        CudaMatrixRow<float, 64> *d_comp_likelihoods = comp_likelihoods.data().get();
        float *d_comp_likelihood_sums = comp_likelihood_sums.data().get();
        float *d_parents_radii = parents_radii.data().get();


        // Create the next level
        float max_radius = 0.0f;
        float *d_max_radius = &max_radius;
        float *d_alpha = &params.alpha;

        cudaMalloc(&d_max_radius, sizeof(float));
        cudaMemcpy(d_max_radius, &max_radius, sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_alpha, sizeof(float));
        cudaMemcpy(d_alpha, &params.alpha, sizeof(float), cudaMemcpyHostToDevice);

        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_parents),
                         [d_components, d_parents, d_parents_radii, d_alpha, d_max_radius]
                                 __device__(unsigned int idx) {
                             unsigned int parent_idx = d_components.parents_indices[idx];
                             d_parents.means[idx] = d_components.means[parent_idx];
                             d_parents.covs[idx] = d_components.covs[parent_idx];
                             d_parents.nvars[idx] = d_components.nvars[parent_idx];

                             float alpha = *d_alpha;
                             d_parents_radii[idx] =
                                     alpha * sqrtf(max_eigenvalues_radius(d_components.covs[parent_idx]));
                             atomicMaxFloat(d_max_radius, d_parents_radii[idx]);
                         });

        //TODO this fails... out of bounds access...
        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_parents),
                         [d_components, d_parents, d_comp_likelihood_sums, d_comp_likelihoods,
                                 d_parents_children_indices, d_max_radius, d_alpha] __device__(unsigned int idx) {
                             unsigned int parent_idx = d_components.parents_indices[idx];

                             unsigned int radius_query_indices[32];
                             const vec3 &mean_parent = d_parents.means[idx];
                             const mat3 &cov_parent = d_parents.covs[idx];
                             const float weight_parent = d_parents.weights[idx];
                             const mat3 cov_parent_inv = cov_parent.inverse();

                             float &comp_likelihood_sum = d_comp_likelihood_sums[parent_idx];
                             auto &comp_likelihood = d_comp_likelihoods[parent_idx];
                             auto &parent_children_indices = d_parents_children_indices[idx];


                             auto nn = query_device(d_parents.d_bvh_means, overlaps_sphere(mean_parent, *d_max_radius),
                                                    radius_query_indices, 32);

                             const float max_likelihood = 1e8f;
                             const float min_likelilhood = FLT_MIN;
                             const float alpha = *d_alpha;
                             const float alpha2 = alpha * alpha;
                             for (int i = 0; i < nn; ++i) {
                                 unsigned int child_idx = radius_query_indices[i];
                                 parent_children_indices.data[i] = child_idx;

                                 const vec3 &child_mean = d_components.means[child_idx];
                                 const mat3 &child_cov = d_components.covs[child_idx];
                                 const float child_weight = d_components.weights[child_idx];

                                 const mat3 child_cov_inv = child_cov.inverse();

                                 float kld = kullback_leibler_divergence(child_mean, child_cov, child_cov_inv,
                                                                         mean_parent,
                                                                         cov_parent, cov_parent_inv);

                                 if (kld > alpha2 * 0.5f) {
                                     continue;
                                 }

                                 if (d_components.is_parent[child_idx] && parent_idx != child_idx) {
                                     continue;
                                 }

                                 float likelihood = hem_likelihood(mean_parent, cov_parent, child_mean,
                                                                   child_cov, child_weight);
                                 float wL_si = weight_parent *
                                               fmaxf(min_likelilhood, fminf(likelihood, max_likelihood));

                                 comp_likelihood.data[child_idx] = wL_si;
                                 atomicAdd(&comp_likelihood_sum, wL_si);
                             }
                         });


        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_parents),
                         [d_components, d_parents, d_parents_children_indices, d_comp_likelihoods, d_comp_likelihood_sums] __device__(
                                 unsigned int idx) {
                             unsigned int parent_idx = d_components.parents_indices[idx];

                             auto &parent_children_indices = d_parents_children_indices[idx];
                             auto &likelihoods = d_comp_likelihoods[parent_idx];

                             float *comp_likelihood_sum = d_comp_likelihood_sums;

                             float w_s = 0.0f;
                             vec3 sumµ_i = vec3::constant(0);
                             mat3 sumcov_i = mat3::constant(0);
                             vec3 resultant = vec3::constant(0);
                             float nvar = 0.0f;

                             for (int i = 0; i < 32; ++i) {
                                 unsigned int child_idx = parent_children_indices.data[i];

                                 if (comp_likelihood_sum[child_idx] == 0) {
                                     continue;
                                 }

                                 const vec3 &child_mean = d_components.means[child_idx];
                                 const mat3 &child_cov = d_components.covs[child_idx];
                                 const float child_weight = d_components.weights[child_idx];
                                 const vec3 &child_nvar = d_components.nvars[child_idx];

                                 float r_cp = likelihoods.data[i] / comp_likelihood_sum[child_idx];
                                 float w = r_cp * child_weight;

                                 float c_nvar = child_nvar.length();
                                 vec3 c_normal = child_nvar / c_nvar;

                                 if (c_normal.dot(d_components.nvars[parent_idx]) < 0.0f) {
                                     c_normal = -c_normal;
                                 }

                                 w_s += w;
                                 sumµ_i = sumµ_i + w * child_mean;
                                 vec3 diff = child_mean - d_components.means[parent_idx];
                                 sumcov_i = sumcov_i + w * (child_cov + outer(diff, diff));

                                 resultant = resultant + w * c_normal;
                                 nvar += w * c_nvar;
                             }

                             float inv_w = 1.0f / w_s;
                             vec3 µ_s = inv_w * sumµ_i;
                             vec3 diff = µ_s - d_components.means[parent_idx];
                             mat3 cov_s = inv_w * sumcov_i - outer(diff, diff);
                             cov_s = conditionCov(cov_s);

                             float variance1 = nvar * inv_w;
                             float R = resultant.length();
                             float Rmean = R * inv_w;
                             float variance2 = -2.0f * log(Rmean);
                             vec3 newMeanNormal = resultant / R;


                             d_parents.means[idx] = µ_s;
                             d_parents.covs[idx] = cov_s;
                             d_parents.weights[idx] = w_s;
                             d_parents.nvars[idx] = newMeanNormal * (variance1 + variance2);
                         });

        unsigned int num_children = components.children_indices.size();

        //count orphans, components not adressed by any parent
        thrust::device_vector<unsigned int> orphans_indices(num_children);
        auto end = thrust::copy_if(components.children_indices.begin(),
                                   components.children_indices.end(),
                                   comp_likelihood_sums.begin(),
                                   orphans_indices.begin(),
        [d_comp_likelihood_sums]
                __device__(unsigned int
        idx) {
            return d_comp_likelihood_sums[idx] == 0.0f;
        }
        );
        orphans_indices.resize(end - orphans_indices.begin());

        HemDeviceData orphans;
        thrust::gather(orphans_indices.begin(), orphans_indices.end(), components.means.begin(),
                       orphans.means.begin());
        thrust::gather(orphans_indices.begin(), orphans_indices.end(), components.covs.begin(),
                       orphans.covs.begin());
        thrust::gather(orphans_indices.begin(), orphans_indices.end(), components.weights.begin(),
                       orphans.weights.begin());
        thrust::gather(orphans_indices.begin(), orphans_indices.end(), components.nvars.begin(),
                       orphans.nvars.begin());

        parents.means.insert(parents.means.end(), orphans.means.begin(), orphans.means.end());
        parents.covs.insert(parents.covs.end(), orphans.covs.begin(), orphans.covs.end());
        parents.weights.insert(parents.weights.end(), orphans.weights.begin(), orphans.weights.end());
        parents.nvars.insert(parents.nvars.end(), orphans.nvars.begin(), orphans.nvars.end());

        parents.h_bvh.clear();
        parents.h_bvh.assign_device(parents.means);

        SeparateParentsChildren(parents);

        cudaFree(d_max_radius);

        return parents;
    }

    HemResult Hem(const std::vector<Vector<float, 3>> &positions, unsigned int levels, unsigned int k) {
        HemResult result;
        std::vector<vec3> ps(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i].x(), positions[i].y(), positions[i].z()};
        }

        HemParams params;
        params.nLevels = levels;
        params.knn = k;

        HemDeviceData data = InitMixture(ps, params);

        for (int i = 0; i < levels; ++i) {
            data = ClusterLevel(data, params);
        }

        thrust::host_vector<vec3> h_means = data.means;
        thrust::host_vector<mat3> h_covs = data.covs;
        thrust::host_vector<float> h_weights = data.weights;
        thrust::host_vector<vec3> h_nvars = data.nvars;

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
/*
        hem mixture;
        hem_params params;
        params.nLevels = levels;
        mixture = hem(ps, params);
        mixture.fit();

        auto h_means = mixture.means_host();
        auto h_covs = mixture.covs_host();
        auto h_weights = mixture.weights_host();
        auto h_nvars = mixture.normal_variance_host();


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
        }*/
        return result;
    }
}