#ifndef LBVH_BVH_CUH
#define LBVH_BVH_CUH

#include "aabb.cuh"
#include "morton_code.cuh"
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <queue>

namespace Bcg::cuda {
    namespace detail {
        struct node {
            std::uint32_t parent_idx; // parent node
            std::uint32_t left_idx; // index of left  child node
            std::uint32_t right_idx; // index of right child node
            std::uint32_t object_idx; // == 0xFFFFFFFF if internal node.
        };

        // a set of pointers to use it on device.
        template<typename Object, bool IsConst>
        struct basic_device_bvh;

        template<typename Object>
        struct basic_device_bvh<Object, false> {
            using node_type = detail::node;
            using index_type = std::uint32_t;
            using object_type = Object;

            unsigned int num_nodes; // (# of internal node) + (# of leaves), 2N+1
            unsigned int num_objects; // (# of leaves), the same as the number of objects

            node_type *nodes;
            aabb *aabbs;
            object_type *objects;
            std::uint32_t *samples;
        };

        template<typename Object>
        struct basic_device_bvh<Object, true> {
            using node_type = detail::node;
            using index_type = std::uint32_t;
            using object_type = Object;

            unsigned int num_nodes; // (# of internal node) + (# of leaves), 2N+1
            unsigned int num_objects; // (# of leaves), the same as the number of objects

            node_type const *nodes;
            aabb const *aabbs;
            object_type const *objects;
            std::uint32_t const *samples;
        };

        template<typename UInt>
        __device__
        inline uint2 determine_range(UInt const *node_code,
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
        inline unsigned int
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

        template<typename Object, bool IsConst, typename UInt>
        void construct_internal_nodes(const basic_device_bvh<Object, IsConst> &self,
                                      UInt const *node_code, const unsigned int num_objects) {
            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator<unsigned int>(0),
                             thrust::make_counting_iterator<unsigned int>(num_objects - 1),
                             [self, node_code, num_objects] __device__(const unsigned int idx) {
                                 self.nodes[idx].object_idx = 0xFFFFFFFF; //  internal nodes

                                 const uint2 ij = determine_range(node_code, num_objects, idx);
                                 const int gamma = find_split(node_code, ij.x, ij.y);

                                 self.nodes[idx].left_idx = gamma;
                                 self.nodes[idx].right_idx = gamma + 1;
                                 if (thrust::min(ij.x, ij.y) == gamma) {
                                     self.nodes[idx].left_idx += num_objects - 1;
                                 }
                                 if (thrust::max(ij.x, ij.y) == gamma + 1) {
                                     self.nodes[idx].right_idx += num_objects - 1;
                                 }
                                 self.nodes[self.nodes[idx].left_idx].parent_idx = idx;
                                 self.nodes[self.nodes[idx].right_idx].parent_idx = idx;
                                 return;
                             });
        }
    } // detail

    template<typename Object>
    struct default_morton_code_calculator {
        default_morton_code_calculator(aabb w) : whole(w) {
        }

        default_morton_code_calculator() = default;

        ~default_morton_code_calculator() = default;

        default_morton_code_calculator(default_morton_code_calculator const &) = default;

        default_morton_code_calculator(default_morton_code_calculator &&) = default;

        default_morton_code_calculator &operator=(default_morton_code_calculator const &) = default;

        default_morton_code_calculator &operator=(default_morton_code_calculator &&) = default;

        __device__ __host__
        inline unsigned int operator()(const Object &, const aabb &box) noexcept {
            auto p = centroid(box);
            p.x -= whole.lower.x;
            p.y -= whole.lower.y;
            p.z -= whole.lower.z;
            p.x /= (whole.upper.x - whole.lower.x);
            p.y /= (whole.upper.y - whole.lower.y);
            p.z /= (whole.upper.z - whole.lower.z);
            return morton_code(p);
        }

        aabb whole;
    };

    template<typename Object>
    using bvh_device = detail::basic_device_bvh<Object, false>;

    template<typename Object>
    using cbvh_device = detail::basic_device_bvh<Object, true>;

    template<typename Object, typename AABBGetter,
        typename MortonCodeCalculator = default_morton_code_calculator<Object> >
    class lbvh {
    public:
        using index_type = std::uint32_t;
        using object_type = Object;
        using node_type = detail::node;
        using aabb_getter_type = AABBGetter;
        using morton_code_calculator_type = MortonCodeCalculator;

    public:
        template<typename InputIterator>
        lbvh(InputIterator first, InputIterator last, bool query_host_enabled = false)
            : objects_h_(first, last), objects_d_(objects_h_),
              query_host_enabled_(query_host_enabled) {
            this->construct();
        }

        lbvh() = default;

        ~lbvh() = default;

        lbvh(const lbvh &) = default;

        lbvh(lbvh &&) = default;

        lbvh &operator=(const lbvh &) = default;

        lbvh &operator=(lbvh &&) = default;

        bool query_host_enabled() const noexcept { return query_host_enabled_; }

        bool &query_host_enabled() noexcept { return query_host_enabled_; }

        void clear() {
            this->objects_h_.clear();
            this->objects_d_.clear();
            this->aabbs_h_.clear();
            this->aabbs_.clear();
            this->nodes_h_.clear();
            this->nodes_.clear();
            this->samples_h_.clear();
            this->samples_.clear();
        }

        template<typename InputIterator>
        void assign_host(InputIterator first, InputIterator last) {
            this->objects_h_.assign(first, last);
            this->objects_d_ = this->objects_h_;
            this->construct();
        }

        void assign_device(const thrust::device_vector<object_type> &d_objects) {
            this->objects_d_ = d_objects;
            this->objects_h_ = this->objects_d_;
            this->construct();
        }

        void assign_device(const object_type *d_objects, const size_t num_objects) {
            this->objects_d_.assign(d_objects, d_objects + num_objects);
            this->objects_h_ = this->objects_d_;
            this->construct();
        }

        bvh_device<object_type> get_device_repr() noexcept {
            return bvh_device<object_type>{
                static_cast<unsigned int>(nodes_.size()),
                static_cast<unsigned int>(objects_d_.size()),
                nodes_.data().get(), aabbs_.data().get(), objects_d_.data().get(), samples_.data().get()
            };
        }

        cbvh_device<object_type> get_device_repr() const noexcept {
            return cbvh_device<object_type>{
                static_cast<unsigned int>(nodes_.size()),
                static_cast<unsigned int>(objects_d_.size()),
                nodes_.data().get(), aabbs_.data().get(), objects_d_.data().get(), samples_.data().get()
            };
        }

        void construct() {
            assert(objects_h_.size() == objects_d_.size());
            if (objects_h_.size() == 0u) { return; }

            const unsigned int num_objects = objects_h_.size();
            const unsigned int num_internal_nodes = num_objects - 1;
            const unsigned int num_nodes = num_objects * 2 - 1;

            // --------------------------------------------------------------------
            // calculate morton code of each points

            const auto inf = std::numeric_limits<float>::infinity();
            aabb default_aabb;
            default_aabb.upper.x = -inf;
            default_aabb.lower.x = inf;
            default_aabb.upper.y = -inf;
            default_aabb.lower.y = inf;
            default_aabb.upper.z = -inf;
            default_aabb.lower.z = inf;

            this->aabbs_.resize(num_nodes, default_aabb);

            thrust::transform(this->objects_d_.begin(), this->objects_d_.end(),
                              aabbs_.begin() + num_internal_nodes, aabb_getter_type());

            const auto aabb_whole = thrust::reduce(
                aabbs_.begin() + num_internal_nodes, aabbs_.end(), default_aabb,
                [] __device__ __host__(const aabb &lhs, const aabb &rhs) {
                    return merge(lhs, rhs);
                });

            thrust::device_vector<unsigned int> morton(num_objects);
            thrust::transform(this->objects_d_.begin(), this->objects_d_.end(),
                              aabbs_.begin() + num_internal_nodes, morton.begin(),
                              morton_code_calculator_type(aabb_whole));


            // --------------------------------------------------------------------
            // sort object-indices by morton code

            thrust::device_vector<unsigned int> indices(num_objects);
            thrust::copy(thrust::make_counting_iterator<index_type>(0),
                         thrust::make_counting_iterator<index_type>(num_objects),
                         indices.begin());
            // keep indices ascending order
            thrust::stable_sort_by_key(morton.begin(), morton.end(),
                                       thrust::make_zip_iterator(
                                           thrust::make_tuple(aabbs_.begin() + num_internal_nodes,
                                                              indices.begin())));


            // --------------------------------------------------------------------
            // check morton codes are unique

            thrust::device_vector<unsigned long long int> morton64(num_objects);
            const auto uniqued = thrust::unique_copy(morton.begin(), morton.end(),
                                                     morton64.begin());

            const bool morton_code_is_unique = (morton64.end() == uniqued);
            if (!morton_code_is_unique) {
                thrust::transform(morton.begin(), morton.end(), indices.begin(),
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

            node_type default_node;
            default_node.parent_idx = 0xFFFFFFFF;
            default_node.left_idx = 0xFFFFFFFF;
            default_node.right_idx = 0xFFFFFFFF;
            default_node.object_idx = 0xFFFFFFFF;
            this->nodes_.resize(num_nodes, default_node);


            thrust::transform(indices.begin(), indices.end(),
                              this->nodes_.begin() + num_internal_nodes,
                              [] __device__(const index_type idx) {
                                  node_type n;
                                  n.parent_idx = 0xFFFFFFFF;
                                  n.left_idx = 0xFFFFFFFF;
                                  n.right_idx = 0xFFFFFFFF;
                                  n.object_idx = idx;
                                  return n;
                              });


            // --------------------------------------------------------------------
            // construct internal nodes

            const auto self = this->get_device_repr();
            if (morton_code_is_unique) {
                const unsigned int *node_code = morton.data().get();
                detail::construct_internal_nodes(self, node_code, num_objects);
            } else // 64bit version
            {
                const unsigned long long int *node_code = morton64.data().get();
                detail::construct_internal_nodes(self, node_code, num_objects);
            }

            // --------------------------------------------------------------------
            // create AABB for each node by bottom-up strategy

            thrust::device_vector<int> flag_container(num_internal_nodes, 0);
            const auto flags = flag_container.data().get();

            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator<index_type>(num_internal_nodes),
                             thrust::make_counting_iterator<index_type>(num_nodes),
                             [self, flags, inf] __device__(index_type idx) {
                                 unsigned int parent = self.nodes[idx].parent_idx;
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
                                     // thread is the 2nd thread. merge AABB of both children.

                                     const auto lidx = self.nodes[parent].left_idx;
                                     const auto ridx = self.nodes[parent].right_idx;
                                     const auto lbox = self.aabbs[lidx];
                                     const auto rbox = self.aabbs[ridx];
                                     self.aabbs[parent] = merge(lbox, rbox);

                                     assert(self.aabbs[lidx].upper.x != -inf);
                                     assert(self.aabbs[ridx].upper.x != -inf);
                                     // look the next parent...
                                     parent = self.nodes[parent].parent_idx;
                                     __threadfence(); //WTF, i did not expect this to be necessary
                                 }
                                 return;
                             });

            if (this->query_host_enabled_) {
                aabbs_h_ = aabbs_;
                nodes_h_ = nodes_;
            }

            fill_samples();
        }

        void fill_samples() {
            const unsigned int num_objects = objects_h_.size();
            const unsigned int num_internal_nodes = num_objects - 1;
            const unsigned int num_nodes = num_objects * 2 - 1;

            this->samples_.resize(num_nodes, 0xFFFFFFFF);

            thrust::transform(nodes_.begin(), nodes_.end(),
                              this->samples_.begin(),
                              [] __device__(const node_type &n) {
                                  return n.object_idx;
                              });


            thrust::device_vector<int> flag_container(num_internal_nodes, 0);
            const auto flags = flag_container.data().get();

            const auto self = this->get_device_repr();
            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator<index_type>(num_internal_nodes),
                             thrust::make_counting_iterator<index_type>(num_nodes),
                             [self, flags] __device__(index_type idx) {
                                 unsigned int parent = self.nodes[idx].parent_idx;
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
                                     // thread is the 2nd thread. merge AABB of both children.

                                     const auto lidx = self.nodes[parent].left_idx;
                                     const auto ridx = self.nodes[parent].right_idx;

                                     assert(lidx != 0xFFFFFFFF);
                                     assert(lidx != 0xFFFFFFFF);

                                     // -- New Sampling Begin --
                                     const auto lsample_idx = self.samples[lidx];
                                     const auto rsample_idx = self.samples[ridx];

                                     assert(lsample_idx != 0xFFFFFFFF);
                                     assert(rsample_idx != 0xFFFFFFFF);
                                     assert(lsample_idx < self.num_objects);
                                     assert(rsample_idx < self.num_objects);

                                     const glm::vec3 &lpoint = self.objects[lsample_idx];
                                     const glm::vec3 &rpoint = self.objects[rsample_idx];
                                     const glm::vec3 center = centroid(self.aabbs[parent]);
                                     if (distance(lpoint, center) < distance(rpoint, center)) {
                                         self.samples[parent] = lsample_idx;
                                     } else {
                                         self.samples[parent] = rsample_idx;
                                     }
                                     //pick a random sample from the two children
                                     unsigned int hash = parent;
                                     hash ^= hash >> 17;
                                     hash ^= hash << 31;
                                     hash ^= hash >> 8;

                                     bool pick_left = (hash & 1) == 0; // Use least significant bit

                                     if (pick_left) {
                                         self.samples[parent] = lsample_idx;
                                     } else {
                                         self.samples[parent] = rsample_idx;
                                     }
                                     // -- New Sampling End --
                                     // look the next parent...
                                     parent = self.nodes[parent].parent_idx;
                                     __threadfence(); //WTF, i did not expect this to be necessary
                                 }
                                 return;
                             });
            //copy from samples_ to samples_h_
            if (this->query_host_enabled_) {
                samples_h_ = samples_;
            }
        }

        unsigned int compute_num_levels(unsigned int num_objects) {
            // Ensure that host copies of nodes are available
            if (nodes_h_.empty()) {
                throw std::runtime_error("Host copies of nodes are not available.");
            }

            unsigned int max_level = 0;

            // Perform BFS to find the maximum depth
            struct NodeInfo {
                unsigned int node_idx;
                unsigned int current_level;
            };

            std::queue<NodeInfo> node_queue;
            node_queue.push({0, 0}); // Start from the root node at level 0

            while (!node_queue.empty()) {
                NodeInfo current = node_queue.front();
                node_queue.pop();

                max_level = std::max(max_level, current.current_level);

                // Get the current node
                const node_type &node = nodes_h_[current.node_idx];

                // Enqueue left child if it exists
                if (node.left_idx != 0xFFFFFFFF) {
                    node_queue.push({node.left_idx, current.current_level + 1});
                }

                // Enqueue right child if it exists
                if (node.right_idx != 0xFFFFFFFF) {
                    node_queue.push({node.right_idx, current.current_level + 1});
                }
            }

            return max_level;
        }

        void compute_level(unsigned int level, unsigned int max_level, unsigned int num_objects,
                           unsigned int &num_objects_in_level,
                           unsigned int &start_idx) {
            if (num_objects == 0) {
                throw std::invalid_argument("num_objects must be greater than 0");
            }

            if (level >= max_level) {
                throw std::invalid_argument("level must be less than num_levels");
            }

            num_objects_in_level = 1 << level; // 2^level
            start_idx = (1 << level) - 1; // 2^level - 1
        }

        /*thrust::host_vector<std::uint32_t> get_samples(unsigned int level) {
            //TODO test this code!
            // Ensure that host queries are enabled
            if (!this->query_host_enabled_) {
                throw std::runtime_error("Host query is not enabled. Set query_host_enabled_ to true before construction.");
            }

            unsigned int num_objects = objects_h_.size();
            unsigned int num_objects_in_level;
            unsigned int start_idx;

            unsigned int max_level = compute_num_levels(num_objects) - 1;
            level = std::min(level, max_level);
            compute_level(level, max_level, num_objects, num_objects_in_level, start_idx);

            // Extract samples at the level
            thrust::host_vector<std::uint32_t> samples(
                    samples_h_.begin() + start_idx,
                    samples_h_.begin() + start_idx + num_objects_in_level
            );

            return samples;
        }*/

        thrust::host_vector<std::uint32_t> get_samples(unsigned int level) {
            // Ensure that host queries are enabled
            if (!this->query_host_enabled_) {
                throw std::runtime_error(
                    "Host query is not enabled. Set query_host_enabled_ to true before construction.");
            }

            // Ensure that host copies of nodes and samples are available
            if (nodes_h_.empty() || samples_h_.empty()) {
                throw std::runtime_error("Host copies of nodes or samples are not available.");
            }

            thrust::host_vector<std::uint32_t> samples;

            // Perform BFS to find nodes at the specified level
            struct NodeInfo {
                unsigned int node_idx;
                unsigned int current_level;
            };

            std::queue<NodeInfo> node_queue;
            node_queue.push({0, 0}); // Start from the root node at level 0

            unsigned int max_level = 0;

            while (!node_queue.empty()) {
                NodeInfo current = node_queue.front();
                node_queue.pop();

                max_level = std::max(max_level, current.current_level);
                const node_type &node = nodes_h_[current.node_idx];
                if (current.current_level == level || node.object_idx != 0xFFFFFFFF) {
                    // Collect the sample at this node
                    samples.push_back(samples_h_[current.node_idx]);
                    // No need to explore further from this node
                    continue;
                }

                // If the current level is less than the desired level, keep traversing
                if (current.current_level < level) {
                    // Get the current node
                    const node_type &node = nodes_h_[current.node_idx];

                    // Enqueue left child if it exists
                    if (node.left_idx != 0xFFFFFFFF) {
                        node_queue.push({node.left_idx, current.current_level + 1});
                    }

                    // Enqueue right child if it exists
                    if (node.right_idx != 0xFFFFFFFF) {
                        node_queue.push({node.right_idx, current.current_level + 1});
                    }
                }
            }

            // Adjust the level if the requested level exceeds the maximum level
            if (samples.empty()) {
                std::cerr << "Requested level " << level << " exceeds maximum tree depth " << max_level <<
                        ". Collecting samples at maximum level instead." << std::endl;
                level = max_level;
                return get_samples(level); // Recurse with maximum level
            }

            return samples;
        }


        thrust::host_vector<object_type> const &objects_host() const noexcept { return objects_h_; }

        thrust::host_vector<object_type> &objects_host() noexcept { return objects_h_; }

        thrust::host_vector<node_type> const &nodes_host() const noexcept { return nodes_h_; }

        thrust::host_vector<node_type> &nodes_host() noexcept { return nodes_h_; }

        thrust::host_vector<aabb> const &aabbs_host() const noexcept { return aabbs_h_; }

        thrust::host_vector<aabb> &aabbs_host() noexcept { return aabbs_h_; }

    private:
        thrust::host_vector<object_type> objects_h_;
        thrust::device_vector<object_type> objects_d_;
        thrust::host_vector<aabb> aabbs_h_;
        thrust::device_vector<aabb> aabbs_;
        thrust::host_vector<node_type> nodes_h_;
        thrust::device_vector<node_type> nodes_;
        thrust::host_vector<std::uint32_t> samples_h_;
        thrust::device_vector<std::uint32_t> samples_;
        bool query_host_enabled_;
    };
} // lbvh
#endif// LBVH_BVH_CUH
