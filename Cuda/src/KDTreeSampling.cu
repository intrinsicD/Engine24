//
// Created by alex on 04.11.24.
//

#include "Cuda/KDTreeSampling.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "AABB.h"

namespace Bcg {
    struct KDTreeSampler {
        struct NodesHostData{
            thrust::host_vector<std::uint32_t> parent_idx; // parent node
            thrust::host_vector<std::uint32_t> left_idx;   // index of left  child node
            thrust::host_vector<std::uint32_t> right_idx;  // index of right child node
            thrust::host_vector<std::uint32_t> object_idx; // == 0xFFFFFFFF if internal node.
        };

        struct NodesDeviceDate{
            thrust::device_vector<std::uint32_t> parent_idx; // parent node
            thrust::device_vector<std::uint32_t> left_idx;   // index of left  child node
            thrust::device_vector<std::uint32_t> right_idx;  // index of right child node
            thrust::device_vector<std::uint32_t> object_idx; // == 0xFFFFFFFF if internal node.
        };

        struct SamplesHostData{
            thrust::host_vector<std::uint32_t> sample_idx; // == 0xFFFFFFFF if internal node.
        };

        struct SamplesDeviceData{
            thrust::device_vector<std::uint32_t> sample_idx; // == 0xFFFFFFFF if internal node.
        };

        struct AABBHostData{
            thrust::host_vector<AABB> aabbs;
        };

        struct AABBDeviceData{
            thrust::device_vector<AABB> aabbs;
        };

        struct PointsHostData{
            thrust::host_vector<Vector<float, 3>> points;
        };

        struct PointsDeviceData{
            thrust::device_vector<Vector<float, 3>> points;
        };

        struct Node {
            std::uint32_t parent_idx; // parent node
            std::uint32_t left_idx;   // index of left  child node
            std::uint32_t right_idx;  // index of right child node
            std::uint32_t object_idx; // == 0xFFFFFFFF if internal node.
            std::uint32_t sample_idx; // == 0xFFFFFFFF if internal node.
        };

        void construct() {

        }

        SamplingResult sample(unsigned int n_level) {
            //sample points up to level k from the root
            //return the sampled points
            return {};
        }

        PointsHostData points_h;
        PointsDeviceData points_d;

        AABBHostData aabbs_h;
        AABBDeviceData aabbs_d;

        NodesHostData nodes_h;
        NodesDeviceDate nodes_d;

        SamplesHostData samples_h;
        SamplesDeviceData samples_d;
    };

    SamplingResult KDTreeSampling(const std::vector<Vector<float, 3>> &points, unsigned int n_level) {
        //upload data points to gpu
        //build kd-tree sampling datastructure
        //sample points up to level k from the root
        //return the sampled points
        return {};
    }
}