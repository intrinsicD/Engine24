//
// Created by alex on 18.08.24.
//

#include "Cuda/Hem.h"
#include "lbvh.cuh"
#include "mat3.cuh"
#include "math.cuh"
#include "vec_operations.cuh"
#include "gaussian.cuh"
#include "Logger.h"
#include <float.h>
#include <thrust/random.h>
#include <thrust/gather.h>

namespace Bcg::cuda {
    using hbvh = lbvh<vec3, aabb_getter<vec3>>;
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
        return real_symmetric_3x3_eigendecomposition(cov, evecs).z;
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

        printf("Separated components into %d parents and %d children\n", num_parents, num_components - num_parents);
    }

    HemDeviceData InitMixture(const std::vector<vec3> &positions, HemParams params) {
        HemDeviceData components;
        components.means = positions;
        components.covs.resize(positions.size());
        components.weights.resize(positions.size());
        components.nvars.resize(positions.size());
        components.is_parent.resize(positions.size());

        SeparateParentsChildren(components);

        components.h_bvh = hbvh(positions.begin(), positions.end(), false);

        HemDeviceDataPtr d_components = components.get_device_repr();

        thrust::device_vector<float> radii(positions.size(), params.globalInitRadius);
        float *d_radii = radii.data().get();

        if (!params.useGlobalInitRadius) {
            float *d_alpha0 = &params.alpha0;
            cudaMalloc(&d_alpha0, sizeof(float));
            cudaMemcpy(d_alpha0, &params.alpha0, sizeof(float), cudaMemcpyHostToDevice);

            thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                             thrust::make_counting_iterator<std::uint32_t>(components.parents_indices.size()),
                             [d_components, d_radii, d_alpha0, params] __device__(unsigned int idx) {
                                 unsigned int parent_idx = d_components.parents_indices[idx];
                                 unsigned int indices[32];
                                 const vec3 &query_point = d_components.means[parent_idx];
                                 auto nn = query_device(d_components.d_bvh_means, knn(query_point, params.knn),
                                                        distance_calculator(),
                                                        indices, 32);
                                 float radius;
                                 float alpha0 = *d_alpha0;

                                 unsigned int neighbor_idx = indices[1];
                                 radius = (d_components.means[parent_idx] - d_components.means[neighbor_idx]).length();

                                 d_radii[idx] = radius * alpha0;
                             });
        }


        bool *d_useWeightedPotentials = &params.useWeightedPotentials;
        cudaMalloc(&d_useWeightedPotentials, sizeof(bool));
        cudaMemcpy(d_useWeightedPotentials, &params.useWeightedPotentials, sizeof(bool), cudaMemcpyHostToDevice);

        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(d_components.num_components),
                         [d_components, d_radii, d_useWeightedPotentials] __device__(unsigned int idx) {
                             const vec3 &query_point = d_components.means[idx];
                             float radius = d_radii[idx];

                             unsigned int indices[64];
                             auto nn = query_device(d_components.d_bvh_means, overlaps_sphere(query_point, radius),
                                                    indices, 64);
                             nn = nn > 64 ? 64 : nn;
                             const float minus_16_over_h2 = -16.0f / (radius * radius);

                             float eps = radius * radius * 0.0001f;
                             float density = 0.000001f;

                             vec3 mean = vec3::constant(0);
                             mat3 cov = mat3::identity() * eps;
                             assert(nn > 0);
                             for (unsigned int i = 0; i < nn; ++i) {
                                 unsigned int neighbor_idx = indices[i];
                                 vec3 neighbor = d_components.means[neighbor_idx];
                                 vec3 diff = neighbor - query_point;
                                 cov = cov + outer(diff, diff);
                                 mean = mean + neighbor;
                                 density += expf(minus_16_over_h2 * diff.dot(diff));
                             }
                             float inv_w = 1.0f / fmaxf(nn, 1.0f);
                             vec3 o = mean * inv_w - query_point;
                             cov = cov * inv_w - outer(o, o);

                             d_components.means[idx] = query_point;
                             d_components.covs[idx] = conditionCov(cov);
                             d_components.weights[idx] = *d_useWeightedPotentials ? 1.0f / density : 1.0f;
                             mat3 evecs;
                             real_symmetric_3x3_eigendecomposition(cov, evecs);
                             float initialVar = 0.001f;
                             d_components.nvars[idx] = evecs.col0 * initialVar;
                             //output all written values
                             printf("i: %d nn: %d mean: %f %f %f cov: %f %f %f %f %f %f %f %f %f weight: %f nvar: %f %f %f\n",
                                    idx, nn,
                                    d_components.means[idx].x,
                                    d_components.means[idx].y, d_components.means[idx].z,
                                    d_components.covs[idx].col0.x, d_components.covs[idx].col1.x,
                                    d_components.covs[idx].col2.x,
                                    d_components.covs[idx].col0.y, d_components.covs[idx].col1.y,
                                    d_components.covs[idx].col2.y,
                                    d_components.covs[idx].col0.z, d_components.covs[idx].col1.z,
                                    d_components.covs[idx].col2.z,
                                    d_components.weights[idx],
                                    d_components.nvars[idx].x, d_components.nvars[idx].y, d_components.nvars[idx].z);
                         });
        return components;
    }

    template<typename T, int N>
    struct CudaMatrixRow {
        T data[N];

        __device__ __host__ CudaMatrixRow() : data() {}

        __device__ __host__ static CudaMatrixRow Zeros() {
            CudaMatrixRow row;
            for (int i = 0; i < N; ++i) {
                row.data[i] = 0;
            }
            return row;
        }

        __device__ __host__ T &operator[](int i) {
            return data[i];
        }

        __device__ __host__ const T &operator[](int i) const {
            return data[i];
        }
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

        thrust::device_vector<CudaMatrixRow<unsigned int, 64>> parents_children_indices(num_parents,
                                                                                        CudaMatrixRow<unsigned int, 64>::Zeros());
        thrust::device_vector<CudaMatrixRow<bool, 64>> parents_children_valid(num_parents,
                                                                              CudaMatrixRow<bool, 64>::Zeros());
        thrust::device_vector<float> parents_radii(num_parents, 0);

        thrust::device_vector<CudaMatrixRow<float, 64>> parent_likelihoods(num_parents,
                                                                           CudaMatrixRow<float, 64>::Zeros());
        thrust::device_vector<float> comp_likelihood_sums(num_components, 0);

        thrust::device_vector<unsigned int> nn_found(num_parents, 0);
        CudaMatrixRow<unsigned int, 64> *d_parents_children_indices = parents_children_indices.data().get();
        CudaMatrixRow<bool, 64> *d_parents_children_valid = parents_children_valid.data().get();
        CudaMatrixRow<float, 64> *d_parents_likelihoods = parent_likelihoods.data().get();
        float *d_comp_likelihood_sums = comp_likelihood_sums.data().get();
        float *d_parents_radii = parents_radii.data().get();
        unsigned int *d_nn_found = nn_found.data().get();


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
                         [d_components, d_parents_radii, d_alpha, d_max_radius]
                                 __device__(unsigned int s_) {
                             unsigned int parent_idx = d_components.parents_indices[s_];

                             float alpha = *d_alpha;
                             float radius = alpha * sqrtf(max_eigenvalues_radius(d_components.covs[parent_idx]));

                             d_parents_radii[s_] = radius;
                             atomicMaxFloat(d_max_radius, radius);
                             printf("Parent %d: radius: %f\n", s_, d_parents_radii[s_]);
                         });

        cudaMemcpy(&max_radius, d_max_radius, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Max radius: %f\n", max_radius);
        //TODO this fails... out of bounds access...
        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_parents),
                         [d_components, d_comp_likelihood_sums, d_parents_likelihoods,
                                 d_parents_children_indices, d_parents_children_valid, d_parents_radii, d_alpha, d_nn_found] __device__(
                                 unsigned int s_) {
                             unsigned int parent_idx = d_components.parents_indices[s_];

                             CudaMatrixRow<unsigned int, 64> radius_query_indices;
                             const vec3 &mean_parent = d_components.means[parent_idx];
                             const mat3 &cov_parent = d_components.covs[parent_idx];
                             const float weight_parent = d_components.weights[parent_idx];
                             const mat3 cov_parent_inv = cov_parent.inverse();

                             CudaMatrixRow<float, 64> &children_likelihoods = d_parents_likelihoods[s_];
                             CudaMatrixRow<unsigned int, 64> &children_indices = d_parents_children_indices[s_];
                             CudaMatrixRow<bool, 64> &children_valid = d_parents_children_valid[s_];
                             float *comp_likelihood_sum = d_comp_likelihood_sums;

                             float radius = d_parents_radii[s_];
                             assert(radius > 0.0f);
                             auto nn = query_device(d_components.d_bvh_means, overlaps_sphere(mean_parent, radius),
                                                    radius_query_indices.data, 64);
                             if (nn < 6) {
                                 nn = query_device(d_components.d_bvh_means, knn(mean_parent, 6), distance_calculator(),
                                                   radius_query_indices.data, 32);
                             }

                             assert(nn > 0);
                             nn = nn > 64 ? 64 : nn;
                             const float max_likelihood = 1e8f;
                             const float min_likelilhood = FLT_MIN;
                             const float alpha = *d_alpha;
                             const float alpha2 = alpha * alpha;
                             d_nn_found[s_] = nn;

                             for (unsigned int i = 0; i < nn; ++i) {
                                 unsigned int child_idx = radius_query_indices[i];
                                 children_indices[i] = child_idx;
                                 const vec3 &child_mean = d_components.means[child_idx];
                                 const mat3 &child_cov = d_components.covs[child_idx];
                                 mat3 child_cov_inv = child_cov.inverse();
                                 const float child_weight = d_components.weights[child_idx];

                                 //TODO! check the kullback leibler divergence
                                 float kld = kullback_leibler_divergence(mean_parent, cov_parent,
                                                                         child_mean, child_cov, child_cov_inv);

                                 if (kld > alpha2 * 0.5f) {
                                     continue;
                                 }

                                 if (d_components.is_parent[child_idx] && parent_idx != child_idx) {
                                     continue;
                                 }

                                 children_valid[i] = true;

                                 float likelihood = hem_likelihood(mean_parent, cov_parent, child_mean,
                                                                   child_cov, child_weight);
                                 float wL_si = weight_parent * clamp(likelihood, min_likelilhood, max_likelihood);

                                 children_likelihoods[i] = wL_si;
                                 printf("nn: %d child_idx: %d valid: %d weight_parent: %f Likelihood: %f wL_si: %f comp_likelihoods[i]: %f kld: %f\n",
                                        d_nn_found[s_], children_indices[i], children_valid[i],
                                        weight_parent, likelihood, wL_si, children_likelihoods[i], kld);
                                 //atomicAdd(&comp_likelihood_sum[child_idx], wL_si);
                                 atomicAdd(thrust::raw_pointer_cast(&comp_likelihood_sum[child_idx]), wL_si);
                             }
                         });

        thrust::host_vector<CudaMatrixRow<float, 64>> h_parent_likelihoods = parent_likelihoods;
        thrust::host_vector<float> h_comp_likelihood_sum = comp_likelihood_sums;
        thrust::host_vector<CudaMatrixRow<unsigned int, 64>> h_parents_children_indices = parents_children_indices;

        for (size_t i = 0; i < h_parents_children_indices.size(); ++i) {
            printf("Parent %zu\n", i);
            for (size_t j = 0; j < 64; ++j) {
                printf("Child %d: %f\n", h_parents_children_indices[i][j], h_parent_likelihoods[i][j]);
            }
        }

        for (size_t i = 0; i < h_comp_likelihood_sum.size(); ++i) {
            printf("Sum: %f\n", h_comp_likelihood_sum[i]);
        }
        thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_parents),
                         [d_components, d_parents, d_parents_children_indices, d_parents_children_valid, d_parents_likelihoods, d_comp_likelihood_sums, d_nn_found] __device__(
                                 unsigned int s_) {
                             unsigned int parent_idx = d_components.parents_indices[s_];

                             CudaMatrixRow<unsigned int, 64> &children_indices = d_parents_children_indices[s_];
                             CudaMatrixRow<float, 64> &children_likelihoods = d_parents_likelihoods[s_];
                             CudaMatrixRow<bool, 64> &children_valid = d_parents_children_valid[s_];

                             float w_s = d_components.weights[parent_idx];
                             vec3 sum_mu_i = w_s * d_components.means[parent_idx];
                             mat3 sum_cov_i = w_s * d_components.covs[parent_idx];
                             vec3 resultant = w_s * d_components.nvars[parent_idx];
                             float nvar = resultant.length();
                             unsigned int nn = d_nn_found[s_];
                             assert(nn > 0);
                             //TODO this loop seems to be never used!!!!
                             for (unsigned int i = 0; i < nn; ++i) {
                                 unsigned int child_idx = children_indices[i];

                                 if (child_idx == parent_idx || !children_valid[i] ||
                                     d_comp_likelihood_sums[child_idx] == 0) {
                                     continue;
                                 }

                                 const vec3 &child_mean = d_components.means[child_idx];
                                 const mat3 &child_cov = d_components.covs[child_idx];
                                 const float child_weight = d_components.weights[child_idx];
                                 const vec3 &child_nvar = d_components.nvars[child_idx];

                                 float r_cp = children_likelihoods[i] / d_comp_likelihood_sums[child_idx];
                                 assert(r_cp > 0.0f);
                                 float w = r_cp * child_weight;
                                 assert(w > 0.0f);
                                 float c_nvar = child_nvar.length();
                                 assert(c_nvar > 0.0f);
                                 vec3 c_normal = child_nvar / c_nvar;

                                 if (c_normal.dot(d_components.nvars[parent_idx]) < 0.0f) {
                                     c_normal = -c_normal;
                                 }

                                 w_s += w;
                                 sum_mu_i = sum_mu_i + w * child_mean;
                                 vec3 diff = child_mean - d_components.means[parent_idx];
                                 sum_cov_i = sum_cov_i + w * (child_cov + outer(diff, diff));

                                 resultant = resultant + w * c_normal;
                                 nvar += w * c_nvar;
                             }

                             float inv_w = 1.0f / w_s;
                             vec3 mu_s = inv_w * sum_mu_i;
                             vec3 diff = mu_s - d_components.means[parent_idx];
                             mat3 cov_s = inv_w * sum_cov_i - outer(diff, diff);

                             float variance1 = nvar * inv_w;
                             float R = resultant.length();
                             float Rmean = R * inv_w;
                             float variance2 = 0;

                             if (nn > 0) {
                                 cov_s = conditionCov(cov_s);
                             } else {
                                 variance2 = -2.0f * log(Rmean);
                             }

                             vec3 newMeanNormal = resultant / R;

                             d_parents.means[s_] = mu_s;
                             d_parents.covs[s_] = cov_s;
                             d_parents.weights[s_] = w_s;
                             d_parents.nvars[s_] = newMeanNormal * (variance1 + variance2);

                             printf("parent: %d nn: %d mean: %f %f %f cov: %f %f %f %f %f %f %f %f %f weight: %f nvar: %f %f %f\n",
                                    s_, nn,
                                    d_parents.means[s_].x, d_parents.means[s_].y, d_parents.means[s_].z,
                                    d_parents.covs[s_].col0.x, d_parents.covs[s_].col1.x, d_parents.covs[s_].col2.x,
                                    d_parents.covs[s_].col0.y, d_parents.covs[s_].col1.y, d_parents.covs[s_].col2.y,
                                    d_parents.covs[s_].col0.z, d_parents.covs[s_].col1.z, d_parents.covs[s_].col2.z,
                                    d_parents.weights[s_],
                                    d_parents.nvars[s_].x, d_parents.nvars[s_].y, d_parents.nvars[s_].z);
                         });

        unsigned int num_children = components.children_indices.size();

        //count orphans, components not adressed by any parent
        thrust::device_vector<unsigned int> orphans_indices(num_children);
        auto end = thrust::copy_if(components.children_indices.begin(),
                                   components.children_indices.end(), orphans_indices.begin(),
        [d_comp_likelihood_sums]
                __device__(unsigned int
        child_idx) {
            return d_comp_likelihood_sums[child_idx] == 0.0f;
        }
        );

        orphans_indices.resize(end - orphans_indices.begin());

        HemDeviceData orphans;
        orphans.means.resize(orphans_indices.size());
        orphans.covs.resize(orphans_indices.size());
        orphans.weights.resize(orphans_indices.size());
        orphans.nvars.resize(orphans_indices.size());
        orphans.is_parent.resize(orphans_indices.size());

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

        assert(orphans_indices.size() < components.means.size());
        assert(parents.means.size() < components.means.size());

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

        return result;
    }
}