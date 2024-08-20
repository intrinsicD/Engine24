//
// Created by alex on 20.08.24.
//

#ifndef ENGINE24_BVH_DEVICE_CUH
#define ENGINE24_BVH_DEVICE_CUH

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

namespace Bcg::cuda::bvh {
    namespace detail {

    }

    template<typename Object, typename Node, typename AABB>
    struct host_data {
        using object_type = Object;
        using node_type = Node;
        using aabb_type = AABB;

        thrust::host_vector<Object> objects;
        thrust::host_vector<node_type> nodes;
        thrust::host_vector<aabb_type> aabb;
    };


    template<typename Object, typename Node, typename AABB>
    struct device_data {
        using object_type = Object;
        using node_type = Node;
        using aabb_type = AABB;

        thrust::device_vector<Object> objects;
        thrust::device_vector<node_type> nodes;
        thrust::device_vector<aabb_type> aabb;
    };

    template<typename Object, typename Node, typename AABB>
    device_data<Object, Node, AABB> get_device_data(const host_data <Object, Node, AABB> &host_data) {
        return {host_data.objects, host_data.nodes, host_data.aabb};
    }

    template<typename Object, typename Node, typename AABB>
    host_data<Object, Node, AABB> get_host_data(const device_data <Object, Node, AABB> &device_data) {
        return {device_data.objects, device_data.nodes, device_data.aabb};
    }

    template<typename Object, typename Node, typename AABB, bool IsConst>
    struct device_ptrs;

    template<typename Object, typename Node, typename AABB>
    struct device_ptrs<Object, Node, AABB, false> {
        using object_type = Object;
        using node_type = Node;
        using aabb_type = AABB;

        unsigned int num_nodes;
        unsigned int num_objects;

        object_type *objects = nullptr;
        node_type *nodes = nullptr;
        aabb_type *aabb = nullptr;
    };

    template<typename Object, typename Node, typename AABB>
    struct device_ptrs<Object, Node, AABB, true> {
        using object_type = Object;
        using node_type = Node;
        using aabb_type = AABB;

        unsigned int num_nodes;
        unsigned int num_objects;


        const object_type *objects;
        const node_type *nodes;
        const aabb_type *aabb;
    };

    template<typename Object, typename Node, typename AABB>
    device_ptrs<Object, Node, AABB, false> get_device_ptrs(device_data < Object, Node, AABB > &device_data) {
        return {static_cast<unsigned int>(device_data.nodes.size()),
                static_cast<unsigned int>(device_data.objects.size()),
                device_data.objects.data().get(),
                device_data.nodes.data().get(),
                device_data.aabb.data().get()};
    }

    template<typename Object, typename Node, typename AABB>
    device_ptrs<Object, Node, AABB, true> get_device_ptrs(const device_data <Object, Node, AABB> &device_data) {
        return {static_cast<unsigned int>(device_data.nodes.size()),
                static_cast<unsigned int>(device_data.objects.size()),
                device_data.objects.data().get(),
                device_data.nodes.data().get(),
                device_data.aabb.data().get()};
    }

    template<typename AABB, typename Object>
    struct getter;

    template<typename AABB>
    struct getter<AABB, vec3> {
        __host__ __device__
        AABB operator()(const vec3 &v) const {
            return AABB(v, v);
        }
    };

    template<typename Object>
    struct merger;

    template<>
    struct merger<aabb> {
        __host__ __device__
        aabb operator()(const aabb &lhs, const aabb &rhs) const {
            aabb merged;
            merged.upper.x = ::fmaxf(lhs.upper.x, rhs.upper.x);
            merged.upper.y = ::fmaxf(lhs.upper.y, rhs.upper.y);
            merged.upper.z = ::fmaxf(lhs.upper.z, rhs.upper.z);
            merged.lower.x = ::fminf(lhs.lower.x, rhs.lower.x);
            merged.lower.y = ::fminf(lhs.lower.y, rhs.lower.y);
            merged.lower.z = ::fminf(lhs.lower.z, rhs.lower.z);
            return merged;
        }
    };

    template<typename Object, typename Container>
    struct factory;

    template<>
    struct factory<aabb, thrust::host_vector<vec3>> {
        aabb operator()(const thrust::host_vector<vec3> &objects) const {
            return thrust::transform_reduce(objects.begin(), objects.end(),
                                            getter<aabb, vec3>(),
                                            aabb(),
                                            merger<aabb>());
        }
    };

    template<>
    struct factory<aabb, thrust::device_vector<vec3>> {
        aabb operator()(const thrust::device_vector<vec3> &objects) const {
            aabb whole;
            return thrust::transform_reduce(thrust::device, objects.begin(), objects.end(),
                                            getter<aabb, vec3>(),
                                            whole,
                                            merger<aabb>());
        }
    };

    template<typename AABB, typename Object>
    struct default_morton_code_calculator {
        AABB whole;
    };


    template<typename Object, typename Node, typename AABB>
    void construct(device_data < Object, Node, AABB > &kdtree) {
        using object_type = Object;
        using node_type = Node;
        using aabb_type = AABB;

        const auto whole = factory<aabb_type, decltype(kdtree.objects)>()(kdtree.objects);


    }
}

#endif //ENGINE24_BVH_DEVICE_CUH
