//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_HEM_MIXTURE_CUH
#define ENGINE24_HEM_MIXTURE_CUH

#include <float.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include "vec3.cuh"
#include "mat3.cuh"
#include "lbvh.cuh"
#include "math.cuh"
#include "vec_operations.cuh"
#include "gaussian.cuh"

namespace Bcg::cuda {
    using hbvh = lbvh<vec3, aabb_getter>;
    using dbvh = bvh_device<vec3>;
    using cdbvh = cbvh_device<vec3>;

    namespace hem_detail {
        struct distance_calculator {
            __device__ __host__
            float operator()(const vec3 &point, const vec3 &object) const

            noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };

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

        template<bool IsConst>
        struct basic_device_hem;

        template<>
        struct basic_device_hem<false> {
            vec3 *means;
            mat3 *covs;
            float *weights;
            vec3 *nvars;
            bool *is_parent;
            unsigned int *parents_indices;
            unsigned int *children_indices;

            dbvh d_bvh_means;

            unsigned int num_components;
            unsigned int num_parents;
            unsigned int num_children;
        };

        template<>
        struct basic_device_hem<true> {
            const vec3 *means;
            const mat3 *covs;
            const float *weights;
            const vec3 *nvars;
            const bool *is_parent;
            const unsigned int *parents_indices;
            const unsigned int *children_indices;

            cdbvh d_bvh_means;

            unsigned int num_components;
            unsigned int num_parents;
            unsigned int num_children;
        };

        template<typename T, int N>
        struct CudaMatrixRow {
            T data[N];
        };

        struct hem_device_data {
            thrust::device_vector <vec3> means;
            thrust::device_vector <mat3> covs;
            thrust::device_vector<float> weights;
            thrust::device_vector <vec3> nvars;
            thrust::device_vector<bool> is_parent;
            thrust::device_vector<unsigned int> parents_indices;
            thrust::device_vector<unsigned int> children_indices;

            basic_device_hem<false> get_device_repr() {
                basic_device_hem<false> d_hem;
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
                return d_hem;
            }

            basic_device_hem<true> get_device_repr() const {
                basic_device_hem<true> d_hem;
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
                return d_hem;
            }
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
        uint knn = 10;                        // number of nearest neighbors to use for kNN initialization
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


    class hem {
    public:
        hem(const thrust::host_vector <vec3> &points, hem_params params) : h_positions(points),
                                                                           d_positions(h_positions), params(params) {

        }

        hem() = default;

        ~hem() = default;

        hem_detail::hem_device_data init_mixture() {
            unsigned int num_components = d_positions.size();
            hem_detail::hem_device_data parents;
            parents.means = h_positions;
            parents.covs.resize(num_components);
            parents.nvars.resize(num_components);
            parents.weights.resize(num_components);
            parents.is_parent.resize(num_components);

            h_bvh = hbvh(h_positions.begin(), h_positions.end(), false);

            auto d_bvh = h_bvh.get_device_repr();

            auto d_parents = parents.get_device_repr();


            //----------------------------------------------------------------------------------------------------------
            // Separate components into parents and children randomly (parents are the first num_parents components)
            //----------------------------------------------------------------------------------------------------------

            unsigned int num_parents = num_components / 3;

            thrust::device_vector<unsigned int> component_indices(num_components);
            thrust::sequence(component_indices.begin(), component_indices.end());

            // Step 2: Generate random keys for shuffling
            thrust::device_vector<float> random_keys(num_components);
            unsigned int seed = 1234; // Or any arbitrary seed
            thrust::transform(thrust::device, component_indices.begin(), component_indices.end(), random_keys.begin(),
                              hem_detail::random_key_generator(seed));

            // Step 3: Sort indices by the random keys to shuffle them
            thrust::sort_by_key(random_keys.begin(), random_keys.end(), component_indices.begin());

            // Step 4: Split into parents and children
            thrust::device_vector<unsigned int> parents_indices(component_indices.begin(),
                                                                component_indices.begin() + num_parents);
            thrust::device_vector<unsigned int> children_indices(component_indices.begin() + num_parents,
                                                                 component_indices.end());

            parents.parents_indices = std::move(parents_indices);
            parents.children_indices = std::move(children_indices);

            thrust::for_each(thrust::device,
                             parents.parents_indices.begin(), parents.parents_indices.end(),
            [d_parents]
            __device__(unsigned int
            parent_idx) {
                d_parents.is_parent[parent_idx] = true;
            });

            float max_radius = 0.0f;
            float *d_max_radius = thrust::raw_pointer_cast(&max_radius);
            //knn query to initialize the search radius
            //TODO check if this is sufficient or if we have to do it like in the original implementation
            thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                             thrust::make_counting_iterator<std::uint32_t>(parents.parents_indices.size()),
                             [d_parents, d_bvh, d_max_radius, this] __device__(unsigned int idx) {
                                 unsigned int parent_idx = d_parents.parents_indices[idx];
                                 unsigned int indices[32];
                                 const vec3 &query_point = d_parents.means[parent_idx];
                                 auto nn = query_device(d_bvh, knn(query_point, params.knn),
                                                        hem_detail::distance_calculator(),
                                                        indices, 32);
                                 for (int i = 0; i < nn; ++i) {
                                     vec3 neighbor = d_parents.means[indices[i]];
                                     float radius = (neighbor - query_point).length();
                                     atomicMaxFloat(d_max_radius, radius);
                                 }
                             });


            thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                             thrust::make_counting_iterator<std::uint32_t>(num_components),
                             [d_parents, d_bvh, d_max_radius, this] __device__(unsigned int idx) {
                                 const vec3 &query_point = d_parents.means[idx];
                                 float r = *d_max_radius * params.alpha0;
                                 unsigned int indices[32];
                                 auto nn = query_device(d_bvh, overlaps_sphere(query_point, r),
                                                        indices, 32);
                                 const float minus_16_over_h2 = -16.0f / (r * r);

                                 float eps = r * r * 0.0001f;
                                 float density = 0.000001f;

                                 vec3 mean = vec3::constant(0);
                                 mat3 cov = mat3::identity() * eps;
                                 for (int i = 0; i < nn; ++i) {
                                     vec3 neighbor = d_parents.means[indices[i]];
                                     vec3 diff = neighbor - query_point;
                                     cov = cov + outer(diff, diff);
                                     mean = mean + neighbor;
                                     density += expf(minus_16_over_h2 * diff.dot(diff));
                                 }

                                 // setup component
                                 float inv_w = 1.0f / nn;        // size > 0 ensured
                                 d_parents.means[idx] = query_point;
                                 vec3 o = mean * inv_w - query_point;
                                 cov = cov * inv_w - outer(o, o);    // consider parent pos centralization
                                 d_parents.covs[idx] = conditionCov(cov);
                                 d_parents.weights[idx] = params.useWeightedPotentials ? 1.0f / density : 1.0f;

                                 // Compute the initial normal and set initial normal variance of this point cluster
                                 // the normal variance is encoded in the length of the normal vector
                                 mat3 evectors;
                                 real_symmetric_3x3_eigendecomposition(cov, &evectors);

                                 // for the initial normal variance, 0.0f should theoretically also work, but since we are encoding the var
                                 // in the length of the normal (to save a channel), thats not possible in our implementation
                                 float initialVar = 0.001f;
                                 d_parents.nvars[idx] = evectors.col0 * initialVar;
                             });
            return parents;
        }

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

        hem_detail::hem_device_data cluster_level(hem_detail::hem_device_data &components) {
            auto d_components = components.get_device_repr();

            auto num_components = d_components.num_components;
            auto num_parents = d_components.num_parents;
            auto num_children = d_components.num_children;

            //----------------------------------------------------------------------------------------------------------
            // Prepare the next level
            //----------------------------------------------------------------------------------------------------------

            hem_detail::hem_device_data parents;
            parents.means.resize(num_parents);
            parents.covs.resize(num_parents);
            parents.nvars.resize(num_parents);
            parents.weights.resize(num_parents);
            parents.is_parent.resize(num_parents);

            auto d_parents = parents.get_device_repr();

            thrust::device_vector <hem_detail::CudaMatrixRow<unsigned int, 32>> parents_children_indices(num_parents);
            thrust::device_vector <hem_detail::CudaMatrixRow<float, 32>> comp_likelihoods(num_components);
            thrust::device_vector<float> comp_likelihood_sums(num_components, 0);
            thrust::device_vector<float> parents_radii(num_parents);

            hem_detail::CudaMatrixRow<unsigned int, 32> *d_parents_children_indices = parents_children_indices.data().get();
            hem_detail::CudaMatrixRow<float, 32> *d_comp_likelihoods = comp_likelihoods.data().get();
            float *d_comp_likelihood_sums = comp_likelihood_sums.data().get();
            float *d_parents_radii = parents_radii.data().get();


            // Create the next level
            float max_radius = 0.0f;
            float *d_max_radius = thrust::raw_pointer_cast(&max_radius);
            float *d_alpha = &params.alpha;
            thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                             thrust::make_counting_iterator<std::uint32_t>(num_parents),
                             [d_components, d_parents, d_parents_radii, d_alpha, d_max_radius, this]
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

            dbvh d_bvh = h_bvh.get_device_repr();

            thrust::for_each(thrust::make_counting_iterator<std::uint32_t>(0),
                             thrust::make_counting_iterator<std::uint32_t>(num_parents),
                             [d_components, d_parents, d_comp_likelihood_sums, d_comp_likelihoods,
                                     d_parents_children_indices, d_bvh, d_max_radius, d_alpha,
                                     this] __device__(unsigned int idx) {
                                 unsigned int parent_idx = d_components.parents_indices[idx];

                                 unsigned int radius_query_indices[32];
                                 const vec3 &mean_parent = d_parents.means[idx];
                                 const mat3 &cov_parent = d_parents.covs[idx];
                                 const float weight_parent = d_parents.weights[idx];
                                 const mat3 cov_parent_inv = cov_parent.inverse();

                                 float &comp_likelihood_sum = d_comp_likelihood_sums[parent_idx];
                                 auto &comp_likelihood = d_comp_likelihoods[parent_idx];
                                 auto &parent_children_indices = d_parents_children_indices[idx];


                                 auto nn = query_device(d_bvh, overlaps_sphere(mean_parent, *d_max_radius),
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

                                     comp_likelihood.data[i] = wL_si;
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

            //count orphans, components not adressed by any parent
            thrust::device_vector<unsigned int> orphans_indices(num_children);
            auto
            end = thrust::copy_if(components.children_indices.begin(),
                                  components.children_indices.end(),
                                  comp_likelihood_sums.begin(),
                                  orphans_indices.begin(),
            [d_comp_likelihood_sums] __device__ (unsigned int
            idx) {
                return d_comp_likelihood_sums[idx] == 0.0f;
            });
            orphans_indices.resize(end - orphans_indices.begin());

            hem_detail::hem_device_data orphans;
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

            h_bvh.clear();
            h_bvh.assign_device(parents.means);

            //----------------------------------------------------------------------------------------------------------
            // Separate components into parents and children randomly (parents are the first num_parents components)
            //----------------------------------------------------------------------------------------------------------

            num_components = parents.means.size();

            thrust::device_vector<unsigned int> component_indices(num_components);
            thrust::sequence(component_indices.begin(), component_indices.end());

            // Step 2: Generate random keys for shuffling
            thrust::device_vector<float> random_keys(num_components);
            unsigned int seed = 1234; // Or any arbitrary seed
            thrust::transform(thrust::device, component_indices.begin(), component_indices.end(), random_keys.begin(),
                              hem_detail::random_key_generator(seed));

            // Step 3: Sort indices by the random keys to shuffle them
            thrust::sort_by_key(random_keys.begin(), random_keys.end(), component_indices.begin());

            // Step 4: Split into parents and children
            thrust::device_vector<unsigned int> parents_indices(component_indices.begin(),
                                                                component_indices.begin() + num_parents);
            thrust::device_vector<unsigned int> children_indices(component_indices.begin() + num_parents,
                                                                 component_indices.end());

            parents.parents_indices = std::move(parents_indices);
            parents.children_indices = std::move(children_indices);

            thrust::for_each(thrust::device,
                             parents.parents_indices.begin(), parents.parents_indices.end(),
            [d_parents]
            __device__(unsigned int
            parent_idx) {
                d_parents.is_parent[parent_idx] = true;
            });

            return parents;
        }

        void device_to_host(hem_detail::hem_device_data &d_hem) {
            h_means = d_hem.means;
            h_covs = d_hem.covs;
            h_nvars = d_hem.nvars;
            h_weights = d_hem.weights;
        }

        void fit() {
            assert(h_positions.size() == d_positions.size());
            if (h_positions.size() == 0) { return; }

            auto d_next = init_mixture();
            for (size_t i = 0; i < params.nLevels; ++i) {
                d_next = cluster_level(d_next);
            }

            device_to_host(d_next);
        }


        const thrust::host_vector <vec3> &positions_host() const

        noexcept { return h_means; }

        const thrust::host_vector <vec3> &means_host() const

        noexcept { return h_means; }

        const thrust::host_vector <mat3> &covs_host() const

        noexcept { return h_covs; }

        const thrust::host_vector <vec3> &normal_variance_host() const

        noexcept { return h_nvars; }

        const thrust::host_vector<float> &weights_host() const

        noexcept { return h_weights; }

    private:
        thrust::host_vector <vec3> h_positions;
        thrust::device_vector <vec3> d_positions;

        hem_params params;

        thrust::host_vector <vec3> h_means;
        thrust::host_vector <mat3> h_covs;
        thrust::host_vector <vec3> h_nvars;
        thrust::host_vector<float> h_weights;

        hem_detail::hem_device_data d_data;

        hbvh h_bvh;
    };
}

#endif //ENGINE24_HEM_MIXTURE_CUH
