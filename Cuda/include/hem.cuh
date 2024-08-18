//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_HEM_MIXTURE_CUH
#define ENGINE24_HEM_MIXTURE_CUH

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include "vec3.cuh"
#include "mat3.cuh"
#include "lbvh.cuh"
#include "math.cuh"
#include "gaussian.cuh"

namespace Bcg::cuda {
    using hbvh = lbvh<vec3, aabb_getter>;
    using dbvh = bvh_device<vec3>;
    using cdbvh = cbvh_device<vec3>;

    struct random_selector {
        float reduction_factor = 1.0 / 3.0;

        __host__ __device__
        random_selector(float reduction_factor) : reduction_factor(reduction_factor) {}

        __host__ __device__
        bool operator()(const int idx) const {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

            rng.discard(idx); // Seed with the index to ensure different random values
            return dist(rng) < reduction_factor; // Select a third of the points
        }
    };

    namespace detail {
        template<bool IsConst>
        struct basic_device_hem;

        template<>
        struct basic_device_hem<false> {
            vec3 *points;
            vec3 *means;
            mat3 *covs;
            vec3 *nvars;
            bool *is_parent;

            dbvh d_bvh_means;

            unsigned int n_clusters;
            unsigned int n_points;
        };

        template<>
        struct basic_device_hem<true> {
            const vec3 *points;
            const vec3 *means;
            const mat3 *covs;
            const vec3 *nvars;
            const bool *is_parent;

            cdbvh d_bvh_means;

            unsigned int n_clusters;
            unsigned int n_points;
        };

        struct hem_device_data {
            thrust::device_vector<vec3> means;
            thrust::device_vector<mat3> covs;
            thrust::device_vector<vec3> nvars;
        };
    }


    struct hem_params {
        float globalInitRadius = 1.0f;        // global initialization kernel radius (applies only if useGlobalInitRadius == true)
        bool useGlobalInitRadius = true;        // use global initialization radius instead of NNDist sampling (more stable for non-uniform point sampling)
        uint nNNDistSamples = 10;            // number of sample points for automatic initialization radius estimation (more is better)
        bool useWeightedPotentials = true;    // if true, performs WLOP-like balancing of the initial Gaussian potentials
        float alpha0 = 2.5f;                    // multiple of nearest neighbor distance to use for initial query radius (<= 2.5f recommended)
        float alpha = 2.5f;                    // multiple of cluster maximum std deviation to use for query radius (<= 2.5f recommended)
        uint nLevels = 5;                    // number of levels to use for hierarchical clustering
        float hemReductionFactor = 3.0f;        // factor by which to reduce the mixture each level (3 is recommended, don't change!)
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

    template<typename InputIterator>
    class hem {
    public:
        using dhem = detail::basic_device_hem<false>;

        hem(InputIterator first, InputIterator last, hem_params params = hem_params()) : params(params),
                                                                                         h_positions(first, last),
                                                                                         d_positions(h_positions) {

        }

        hem() = default;

        ~hem() = default;

        hem(const hem &) = default;

        hem(hem &&) = default;

        hem &operator=(const hem &) = default;

        hem &operator=(hem &&) = default;

        dhem init_mixture();

        __device__ __host__ float max_eigenvalues_radius(const mat3 &cov) {
            return real_symmetric_3x3_eigendecomposition(cov, nullptr).z;
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

        dhem cluster_level(detail::hem_device_data &prev) {
            thrust::device_vector<int> indices(prev.means.size());
            thrust::device_vector<int> selected_indices(prev.means.size());

            // Initialize the indices
            thrust::sequence(indices.begin(), indices.end());

            // Apply the random selector functor with the reduction factor
            random_selector selector(params.hemReductionFactor);
            auto end = thrust::copy_if(indices.begin(), indices.end(), selected_indices.begin(), selector);
            selected_indices.resize(end - selected_indices.begin());

            detail::hem_device_data next;
            next.means.resize(selected_indices.size());
            next.covs.resize(selected_indices.size());
            next.nvars.resize(selected_indices.size());

            thrust::device_vector<float> radius(selected_indices.size());
            // Create the next level
            float max_radius = 0.0f;
            thrust::for_each(selected_indices.begin(), selected_indices.end(), [&](int idx) {
                next.means[idx] = prev.means[idx];
                next.covs[idx] = prev.covs[idx];
                next.nvars[idx] = prev.nvars[idx];

                radius[idx] = params.alpha * sqrtf(max_eigenvalues_radius(prev.covs[idx]));
                atomicMaxFloat(&max_radius, radius[idx]);
            });

            h_bvh.clear();
            h_bvh.assign_device(next.means);
            dbvh d_bvh = h_bvh.get_device_repr();

            thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                             thrust::make_counting_iterator<std::uint32_t>(selected_indices.size()), [&](int idx) {
                        unsigned int indices[32];
                        auto nn = query_device(d_bvh, overlaps_sphere(next.means[idx], max_radius), indices, 32);
                        const vec3 &mean_parent = next.means[idx];
                        const mat3 &cov_parent = next.covs[idx];
                        const mat3 cov_parent_inv = cov_parent.inverse();
                        for (int i = 0; i < nn; ++i) {
                            const vec3 &mean_child = next.means[indices[i]];
                            const mat3 &cov_child = next.covs[indices[i]];
                            const mat3 cov_child_inv = cov_child.inverse();
                            float kld = kullback_leibler_divergence(mean_child, cov_child, cov_child_inv, mean_parent,
                                                                    cov_parent, cov_parent_inv);
                            if(kld > params.alpha * params.alpha * 0.5f){
                                continue;
                            }
                            //TODO continue here!
                        }
                    });


            d_next.means = new vec3[d_next.n_clusters];
            d_next.covs = new mat3[d_next.n_clusters];
            d_next.nvars = new vec3[d_next.n_clusters];
            d_next.is_parent = new bool[d_next.n_clusters];

            d_next.d_bvh_means = dbvh(d_next.means, d_next.n_clusters);

            return d_next;
        }

        void device_to_host(dhem &d_hem) {
            h_means.resize(d_hem.n_clusters);
            h_covs.resize(d_hem.n_clusters);
            h_nvars.resize(d_hem.n_clusters);
            h_is_parent.resize(d_hem.n_clusters);

            thrust::copy(d_hem.means, d_hem.means + d_hem.n_clusters, h_means.begin());
            thrust::copy(d_hem.covs, d_hem.covs + d_hem.n_clusters, h_covs.begin());
            thrust::copy(d_hem.nvars, d_hem.nvars + d_hem.n_clusters, h_nvars.begin());
            thrust::copy(d_hem.is_parent, d_hem.is_parent + d_hem.n_clusters, h_is_parent.begin());
        }

        __device__ __host__ void sample_nn_distances(int n_samples);

        void fit() {
            assert(h_positions.size() == d_positions.size());
            if (h_positions.size() == 0) { return; }

            auto d_next = init_mixture();
            for (size_t i = 0; i < params.nLevels; ++i) {
                d_next = cluster_level(d_next);
            }

            device_to_host(d_next);
        }

        __device__ __host__ float hem_likelihood(size_t parend_idx, size_t child_idx);

        const thrust::host_vector<vec3> &positions_host() const noexcept { return h_means; }

        const thrust::host_vector<vec3> &means_host() const noexcept { return h_means; }

        const thrust::host_vector<mat3> &covs_host() const noexcept { return h_covs; }

        const thrust::host_vector<vec3> &normal_variance_host() const noexcept { return h_nvars; }

    private:
        hem_params params;

        thrust::host_vector<vec3> h_positions;
        thrust::device_vector<vec3> d_positions;

        thrust::host_vector<vec3> h_means;
        thrust::host_vector<mat3> h_covs;
        thrust::host_vector<vec3> h_nvars;

        detail::hem_device_data d_data;

        hbvh h_bvh;
    };
}

#endif //ENGINE24_HEM_MIXTURE_CUH
