//
// Created by alex on 16.08.24.
//

#include "LocalGaussians.h"
#include "Gaussian.h"
#include "lbvh.cuh"

#include "thrust/device_vector.h"
#include "thrust/for_each.h"

namespace Bcg {
    using hbvh = lbvh::bvh<float, float4, lbvh::aabb_getter>;
    using dbvh = lbvh::bvh_device<float, float4>;

    struct LocalGaussiansDevicePtr {
        thrust::device_ptr<float4> d_positions;
        thrust::device_ptr<float4> d_means;
        thrust::device_ptr<Matrix<float, 3, 3>> d_covs;
        dbvh d_bvh;
    };

    struct LocalGaussiansDeviceData {
        thrust::device_vector<float4> positions;
        thrust::device_vector<float4> means;
        thrust::device_vector<Matrix<float, 3, 3>> covs;
        hbvh bvh;

        LocalGaussiansDevicePtr GetDevicePtr() {
            return {positions.data(),
                    means.data(),
                    covs.data(),
                    bvh.get_device_repr()};
        }
    };

    LocalGaussiansDeviceData SetupLocalGaussiansDeviceData(const thrust::host_vector<float4> h_positions) {
        return {h_positions,
                thrust::device_vector<float4>(h_positions.size()),
                thrust::device_vector<Matrix<float, 3, 3>>(h_positions.size()),
                hbvh(h_positions.begin(), h_positions.end(), true)
        };
    }

    LocalGaussiansResult DeviceToHost(const LocalGaussiansDeviceData &data) {
        // Determine the number of objects
        size_t num_objects = data.positions.size();

        // Step 1: Copy means from device to host
        thrust::host_vector<float4> h_means = data.means;

        // Convert means from float4 to Vector<float, 3>
        std::vector<Vector<float, 3>> means(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            means[i] = Vector<float, 3>(h_means[i].x, h_means[i].y, h_means[i].z);
        }

        // Step 2: Copy covariance matrices from device to host
        thrust::host_vector<Matrix<float, 3, 3>> h_covs = data.covs;

        // Return the final result as LocalGaussiansResult
        LocalGaussiansResult result;
        result.means = means;
        result.covs = std::vector<Matrix<float, 3, 3>>(h_covs.data(), h_covs.data() + h_covs.size());
        return result;
    }

    LocalGaussiansResult
    LocalGaussians(const std::vector<Vector<float, 3>> &points, const std::vector<unsigned int> &labels) {
        return {};
    }

    struct distance_calculator {
        __device__ __host__
        float operator()(const float4 point, const float4 object) const noexcept {
            return (point.x - object.x) * (point.x - object.x) +
                   (point.y - object.y) * (point.y - object.y) +
                   (point.z - object.z) * (point.z - object.z);
        }
    };

    void ComputeCovs(const dbvh &d_bvh,
                     thrust::device_ptr<float4> d_positions,
                     thrust::device_ptr<float4> d_means,
                     thrust::device_ptr<Matrix<float, 3, 3>> d_covs,
                     size_t num_objects, int k) {

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_objects),
                         [d_bvh, d_positions, d_means, d_covs, k] __device__(
                                 std::uint32_t idx) {
                             const float4 &query = d_positions[idx];
                             unsigned int indices[32];
                             const auto num_found = lbvh::query_device(d_bvh, lbvh::knn(query, k),
                                                                       distance_calculator(), indices);

                             Matrix<float, 3, 3> &cov = *thrust::raw_pointer_cast(d_covs + idx);
                             float4 &mean = *thrust::raw_pointer_cast(d_means + idx);
                             mean = {0.0f, 0.0f, 0.0f, 0.0f};
                             for (int i = 0; i < num_found; ++i) {
                                 const float4 &object = d_positions[indices[i]];
                                 mean.x += object.x;
                                 mean.y += object.y;
                                 mean.z += object.z;
                             }

                             mean.x /= num_found;
                             mean.y /= num_found;
                             mean.z /= num_found;

                             cov = Matrix<float, 3, 3>::Zero();
                             for (int i = 0; i < num_found; ++i) {
                                 const float4 &object = d_positions[indices[i]];
                                 const float4 diff = {object.x - mean.x, object.y - mean.y, object.z - mean.z, 0.0f};
                                 cov(0, 0) += diff.x * diff.x;
                                 cov(0, 1) += diff.x * diff.y;
                                 cov(0, 2) += diff.x * diff.z;
                                 cov(1, 0) += diff.y * diff.x;
                                 cov(1, 1) += diff.y * diff.y;
                                 cov(1, 2) += diff.y * diff.z;
                                 cov(2, 0) += diff.z * diff.x;
                                 cov(2, 1) += diff.z * diff.y;
                                 cov(2, 2) += diff.z * diff.z;
                             }

                             cov /= num_found;
                         }
        );
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }

    LocalGaussiansResult LocalGaussians(const std::vector<Vector<float, 3>> &points, int k) {
        const size_t num_objects = points.size();
        thrust::host_vector<float4> h_positions(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            h_positions[i] = {points[i].x(), points[i].y(), points[i].z(), 1.0f};
        }

        LocalGaussiansDeviceData data = SetupLocalGaussiansDeviceData(h_positions);
        LocalGaussiansDevicePtr ptr = data.GetDevicePtr();

        ComputeCovs(ptr.d_bvh, ptr.d_positions, ptr.d_means, ptr.d_covs, num_objects, std::min(k, 32));
        return DeviceToHost(data);
    }
}