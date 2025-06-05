//
// Created by alex on 20.08.24.
//

#ifndef ENGINE24_BVH_DEVICE_CUH
#define ENGINE24_BVH_DEVICE_CUH

#include "aabb_utils.cuh"
#include "morton_code.cuh"
#include <thrust/swap.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

namespace Bcg::cuda::bvh {
    namespace detail {
        struct node {
            std::uint32_t parent_idx; // parent node
            std::uint32_t left_idx; // index of left  child node
            std::uint32_t right_idx; // index of right child node
            std::uint32_t object_idx; // == 0xFFFFFFFF if internal node.
        };

        template<typename Object, bool IsConst>
        struct basic_device_bvh;

        template<typename Object>
        struct basic_device_bvh<Object, false> {
            using node_type = detail::node;
            using aabb_type = aabb;
            using object_type = Object;

            size_t num_internal_nodes;
            size_t num_nodes;
            size_t num_objects;

            node_type *nodes = nullptr;
            aabb_type *aabbs = nullptr;
            object_type *objects = nullptr;
            aabb_type *aabb_whole = nullptr;
        };

        template<typename Object>
        struct basic_device_bvh<Object, true> {
            using node_type = detail::node;
            using aabb_type = aabb;
            using object_type = Object;

            size_t num_internal_nodes;
            size_t num_nodes;
            size_t num_objects;

            const node_type *nodes = nullptr;
            const aabb_type *aabbs = nullptr;
            const object_type *objects = nullptr;
            const aabb_type *aabb_whole = nullptr;
        };

        template<typename UInt>
        __device__
        uint2 determine_range(UInt const *node_code,
                              const unsigned int num_leaves, unsigned int idx) {
            if (idx == 0) {
                return make_uint2(0, num_leaves - 1);
            }

            // determine direction of the range
            const UInt self_code = node_code[idx];
            const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
            const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
            const int d = (R_delta > L_delta) ? 1 : -1;

            // Compute upper bound for the length of the range

            const int delta_min = thrust::min(L_delta, R_delta);
            int l_max = 2;
            int delta = -1;
            int i_tmp = idx + d * l_max;
            if (0 <= i_tmp && i_tmp < num_leaves) {
                delta = common_upper_bits(self_code, node_code[i_tmp]);
            }
            while (delta > delta_min) {
                l_max <<= 1;
                i_tmp = idx + d * l_max;
                delta = -1;
                if (0 <= i_tmp && i_tmp < num_leaves) {
                    delta = common_upper_bits(self_code, node_code[i_tmp]);
                }
            }

            // Find the other end by binary search
            int l = 0;
            int t = l_max >> 1;
            while (t > 0) {
                i_tmp = idx + (l + t) * d;
                delta = -1;
                if (0 <= i_tmp && i_tmp < num_leaves) {
                    delta = common_upper_bits(self_code, node_code[i_tmp]);
                }
                if (delta > delta_min) {
                    l += t;
                }
                t >>= 1;
            }
            unsigned int jdx = idx + l * d;
            if (d < 0) {
                thrust::swap(idx, jdx); // make it sure that idx < jdx
            }
            return make_uint2(idx, jdx);
        }

        template<typename UInt>
        __device__
        unsigned int
        find_split(UInt const *node_code, const unsigned int first, const unsigned int last) noexcept {
            const UInt first_code = node_code[first];
            const UInt last_code = node_code[last];
            if (first_code == last_code) {
                return (first + last) >> 1;
            }
            const int delta_node = common_upper_bits(first_code, last_code);

            // binary search...
            int split = first;
            int stride = last - first;
            do {
                stride = (stride + 1) >> 1;
                const int middle = split + stride;
                if (middle < last) {
                    const int delta = common_upper_bits(first_code, node_code[middle]);
                    if (delta > delta_node) {
                        split = middle;
                    }
                }
            } while (stride > 1);

            return split;
        }

        template<typename UInt>
        void construct_internal_nodes(detail::node *nodes, UInt const *node_code, const unsigned int num_objects) {
            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator<unsigned int>(0),
                             thrust::make_counting_iterator<unsigned int>(num_objects - 1),
                             [nodes, node_code, num_objects] __device__(const unsigned int idx) {
                                 nodes[idx].object_idx = 0xFFFFFFFF; //  internal nodes

                                 const uint2 ij = determine_range(node_code, num_objects, idx);
                                 const int gamma = find_split(node_code, ij.x, ij.y);

                                 nodes[idx].left_idx = gamma;
                                 nodes[idx].right_idx = gamma + 1;
                                 if (thrust::min(ij.x, ij.y) == gamma) {
                                     nodes[idx].left_idx += num_objects - 1;
                                 }
                                 if (thrust::max(ij.x, ij.y) == gamma + 1) {
                                     nodes[idx].right_idx += num_objects - 1;
                                 }
                                 nodes[nodes[idx].left_idx].parent_idx = idx;
                                 nodes[nodes[idx].right_idx].parent_idx = idx;
                             });
        }
    }

    template<typename Object>
    struct host_data {
        using node_type = detail::node;
        using aabb_type = aabb;
        using object_type = Object;

        thrust::host_vector<node_type> nodes;
        thrust::host_vector<aabb_type> aabbs;
        thrust::host_vector<object_type> objects;
        aabb_type aabb_whole;

        size_t num_internal_nodes() const {
            return objects.size() - 1;
        }

        size_t num_nodes() const {
            return objects.size() * 2 - 1;
        }
    };

    template<typename Object>
    struct device_data {
        using node_type = detail::node;
        using aabb_type = aabb;
        using object_type = Object;

        thrust::device_vector<node_type> nodes;
        thrust::device_vector<aabb_type> aabbs;
        thrust::device_vector<object_type> objects;
        aabb_type aabb_whole;

        size_t num_internal_nodes() const {
            return objects.size() - 1;
        }

        size_t num_nodes() const {
            return objects.size() * 2 - 1;
        }
    };

    template<typename Object>
    device_data<Object> get_device_data(const host_data<Object> &h_lbvh) {
        device_data<Object> d_lbvh;
        d_lbvh.nodes = h_lbvh.nodes;
        d_lbvh.aabbs = h_lbvh.aabbs;
        d_lbvh.objects = h_lbvh.objects;
        d_lbvh.aabb_whole = h_lbvh.aabb_whole;
        return d_lbvh;
    }

    template<typename Object>
    host_data<Object> get_host_data(const device_data<Object> &d_lbvh) {
        host_data<Object> h_lbvh;
        h_lbvh.nodes = d_lbvh.nodes;
        h_lbvh.aabbs = d_lbvh.aabbs;
        h_lbvh.objects = d_lbvh.objects;
        h_lbvh.aabb_whole = d_lbvh.aabb_whole;
        return h_lbvh;
    }

    template<typename Object>
    detail::basic_device_bvh<Object, false> get_device_ptrs(device_data<Object> &d_lbvh) {
        detail::basic_device_bvh<Object, false> d_ptrs;
        d_ptrs.num_internal_nodes = d_lbvh.num_internal_nodes();
        d_ptrs.num_nodes = d_lbvh.num_nodes();
        d_ptrs.num_objects = d_lbvh.objects.size();
        d_ptrs.nodes = d_lbvh.nodes.data().get();
        d_ptrs.aabbs = d_lbvh.aabbs.data().get();
        d_ptrs.objects = d_lbvh.objects.data().get();
        d_ptrs.aabb_whole = &d_lbvh.aabb_whole;
        return d_ptrs;
    }

    template<typename Object>
    detail::basic_device_bvh<Object, true> get_device_ptrs(const device_data<Object> &d_lbvh) {
        detail::basic_device_bvh<Object, true> d_ptrs;
        d_ptrs.num_internal_nodes = d_lbvh.num_internal_nodes();
        d_ptrs.num_nodes = d_lbvh.num_nodes();
        d_ptrs.num_objects = d_lbvh.objects.size();
        d_ptrs.nodes = d_lbvh.nodes.data().get();
        d_ptrs.aabbs = d_lbvh.aabbs.data().get();
        d_ptrs.objects = d_lbvh.objects.data().get();
        d_ptrs.aabb_whole = &d_lbvh.aabb_whole;
        return d_ptrs;
    }

    struct default_morton_code_calculator {
        aabb whole;

        __host__ __device__
        unsigned int operator()(const aabb &box) const noexcept {
            // ✅ Note: const here
            auto p = centroid(box);
            p[0] -= whole.min[0];
            p[1] -= whole.min[1];
            p[2] -= whole.min[2];
            p[0] /= (whole.max[0] - whole.min[0]);
            p[1] /= (whole.max[1] - whole.min[1]);
            p[2] /= (whole.max[2] - whole.min[2]);
            return morton_code(p);
        }
    };

    struct aabb_merger {
        __host__ __device__
        aabb operator()(const aabb &a, const aabb &b) const noexcept {
            return merge(a, b);
        }
    };

    template<typename Object>
    void construct_aabbs_per_object(device_data<Object> &lbvh) {
        const auto num_nodes = lbvh.num_nodes();
        if (lbvh.aabbs.size() != num_nodes) {
            lbvh.aabbs.resize(num_nodes);
        }
        thrust::transform(thrust::device,
                          lbvh.objects.begin(),
                          lbvh.objects.end(),
                          lbvh.aabbs.begin() + lbvh.num_internal_nodes(), aabb_getter<Object>());
    }

    template<typename Object>
    aabb compute_whole_aabb(const device_data<Object> &lbvh) {
        constexpr auto inf = std::numeric_limits<float>::infinity();
        aabb whole{vec3(-inf), vec3(inf)};
        return thrust::reduce(thrust::device,
                              lbvh.aabbs.begin() + lbvh.num_internal_nodes(),
                              lbvh.aabbs.end(),
                              whole,
                              aabb_merger{});
    }

    template<typename Object>
    void compute_morton_codes(const device_data<Object> &lbvh,
                              thrust::device_vector<unsigned int> &morton_codes) {
        thrust::transform(thrust::device,
                          lbvh.aabbs.begin() + lbvh.num_internal_nodes(),
                          lbvh.aabbs.end(),
                          morton_codes.begin(),
                          default_morton_code_calculator{lbvh.aabb_whole});
    }

    //Assumes that the aabbs are already calculated per object and stored in aabbs.begin() + num_internal_nodes to aabbs.end()
    template<typename Object>
    void construct_device(device_data<Object> &lbvh) {
        using index_type = std::uint32_t;

        const unsigned int num_objects = lbvh.objects.size();
        const unsigned int num_internal_nodes = lbvh.num_internal_nodes();
        const unsigned int num_nodes = lbvh.num_nodes();

        // --------------------------------------------------------------------
        // calculate morton code of each points

        if (lbvh.aabbs.empty()) {
            construct_aabbs_per_object(lbvh);
        }

        if (lbvh.aabb_whole.empty()) {
            // calculate whole AABB if it is not given.
            lbvh.aabb_whole = compute_whole_aabb(lbvh);
        }

        thrust::device_vector<unsigned int> morton(num_objects);
        compute_morton_codes<Object>(lbvh, morton);

        // --------------------------------------------------------------------
        // sort object-indices by morton code

        thrust::device_vector<unsigned int> indices(num_objects);
        thrust::copy(thrust::device, thrust::make_counting_iterator<index_type>(0),
                     thrust::make_counting_iterator<index_type>(num_objects),
                     indices.begin());
        // keep indices ascending order
        thrust::stable_sort_by_key(thrust::device, morton.begin(), morton.end(),
                                   thrust::make_zip_iterator(
                                           thrust::make_tuple(lbvh.aabbs.begin() + num_internal_nodes,
                                                              indices.begin())));

        // --------------------------------------------------------------------
        // check morton codes are unique

        thrust::device_vector<unsigned long long int> morton64(num_objects);
        const auto uniqued = thrust::unique_copy(thrust::device,
                                                 morton.begin(),
                                                 morton.end(),
                                                 morton64.begin());

        const bool morton_code_is_unique = (morton64.end() == uniqued);
        if (!morton_code_is_unique) {
            thrust::transform(thrust::device, morton.begin(), morton.end(), indices.begin(),
                              morton64.begin(),
                              [] __device__(const unsigned int m, const unsigned int idx) {
                                  unsigned long long int m64 = m;
                                  m64 <<= 32;
                                  m64 |= idx;
                                  return m64;
                              });
        }

        // --------------------------------------------------------------------
        // construct leaf nodes and aabbs

        detail::node default_node;
        default_node.parent_idx = 0xFFFFFFFF;
        default_node.left_idx = 0xFFFFFFFF;
        default_node.right_idx = 0xFFFFFFFF;
        default_node.object_idx = 0xFFFFFFFF;
        if(lbvh.nodes.size() != num_nodes) {
            lbvh.nodes.resize(num_nodes, default_node);
        }

        thrust::transform(thrust::device, indices.begin(), indices.end(),
                          lbvh.nodes.begin() + num_internal_nodes,
                          [] __device__(const index_type idx) {
                              detail::node n{};
                              n.parent_idx = 0xFFFFFFFF;
                              n.left_idx = 0xFFFFFFFF;
                              n.right_idx = 0xFFFFFFFF;
                              n.object_idx = idx;
                              return n;
                          });

        // --------------------------------------------------------------------
        // construct internal nodes

        const auto self = get_device_ptrs(lbvh);

        if (morton_code_is_unique) {
            const unsigned int *node_code = morton.data().get();
            detail::construct_internal_nodes(self.nodes, node_code, num_objects);
        } else // 64bit version
        {
            const unsigned long long int *node_code = morton64.data().get();
            detail::construct_internal_nodes(self.nodes, node_code, num_objects);
        }

        // --------------------------------------------------------------------
        // create AABB for each node by bottom-up strategy

        thrust::device_vector<int> flag_container(num_internal_nodes, 0);
        const auto flags = flag_container.data().get();


        aabb *__restrict__ aabbs = self.aabbs;
        const detail::node *__restrict__ const_nodes = self.nodes;
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<index_type>(num_internal_nodes),
                         thrust::make_counting_iterator<index_type>(num_nodes),
                         [const_nodes, aabbs, flags] __device__(index_type idx) {
                             unsigned int parent = const_nodes[idx].parent_idx;
                             while (parent != 0xFFFFFFFF) // means idx == 0
                             {
                                 const int old = atomicCAS(flags + parent, 0, 1);
                                 if (old == 0) {
                                     // this is the first thread entered here.
                                     // wait the other thread from the other child node.
                                     return;
                                 }
                                 assert(old == 1);
                                 // here, the flag has already been 1. it means that this
                                 // thread is the 2nd thread. merge AABB of both childlen.

                                 const auto lidx = const_nodes[parent].left_idx;
                                 const auto ridx = const_nodes[parent].right_idx;
                                 const auto lbox = aabbs[lidx];
                                 const auto rbox = aabbs[ridx];
                                 aabbs[parent] = merge(lbox, rbox);

                                 // look the next parent...
                                 parent = const_nodes[parent].parent_idx;
                                 __threadfence();
                                 //WTF, i did not expect this to be necessary, do i really need it here?
                             }
                         });
    }

    //Assumes that the aabbs are already calculated per object and stored in aabbs.begin() + num_internal_nodes to aabbs.end()
    template<typename Object>
    void construct_host(host_data<Object> &lbvh) {
        device_data d_lbvh = get_device_data(lbvh);
        construct_device(d_lbvh);
        lbvh = get_host_data(d_lbvh);
    }
}

#endif //ENGINE24_BVH_DEVICE_CUH
