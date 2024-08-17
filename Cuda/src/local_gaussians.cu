//
// Created by alex on 16.08.24.
//

#include "Cuda/LocalGaussians.h"
#include "lbvh.cuh"
#include "vec_operations.cuh"

#include "thrust/device_vector.h"
#include "thrust/for_each.h"

namespace Bcg::cuda {
    using hbvh = lbvh<vec3, aabb_getter>;
    using dbvh = bvh_device<vec3>;

    struct LocalGaussiansDevicePtr {
        thrust::device_ptr<vec3> d_positions;
        thrust::device_ptr<vec3> d_means;
        thrust::device_ptr<mat3> d_covs;
        dbvh d_bvh;
    };

    struct LocalGaussiansDeviceData {
        thrust::device_vector<vec3> positions;
        thrust::device_vector<vec3> means;
        thrust::device_vector<mat3> covs;
        hbvh bvh;

        LocalGaussiansDevicePtr GetDevicePtr() {
            return {positions.data(),
                    means.data(),
                    covs.data(),
                    bvh.get_device_repr()};
        }
    };

    LocalGaussiansDeviceData SetupLocalGaussiansDeviceData(const thrust::host_vector<vec3> &h_positions) {
        return {h_positions,
                thrust::device_vector<vec3>(h_positions.size()),
                thrust::device_vector<mat3>(h_positions.size()),
                hbvh(h_positions.begin(), h_positions.end(), true)
        };
    }

    LocalGaussiansResult DeviceToHost(const LocalGaussiansDeviceData &data) {
        size_t num_objects = data.positions.size();

        // Preallocate host vectors with the correct size
        thrust::host_vector<vec3> h_means = data.means;
        std::vector<Vector<float, 3>> host_means(h_means.size());
        thrust::transform(h_means.begin(), h_means.end(), host_means.begin(),
                          [] __host__(const vec3 &v) {
                              return Vector<float, 3>{v.x, v.y, v.z};
                          });

        // Copy covariances from device to host and convert to Matrix<float, 3, 3>
        thrust::host_vector<mat3> h_covs = data.covs;
        std::vector<Matrix<float, 3, 3>> host_covs(h_covs.size());
        thrust::transform(h_covs.begin(), h_covs.end(), host_covs.begin(),
                          [] __host__(const mat3 &m) {
                              Matrix<float, 3, 3> mat;
                              mat(0, 0) = m.col0.x;
                              mat(0, 1) = m.col0.y;
                              mat(0, 2) = m.col0.z;
                              mat(1, 0) = m.col1.x;
                              mat(1, 1) = m.col1.y;
                              mat(1, 2) = m.col1.z;
                              mat(2, 0) = m.col2.x;
                              mat(2, 1) = m.col2.y;
                              mat(2, 2) = m.col2.z;
                              return mat;
                          });

        // Return the final result as LocalGaussiansResult
        LocalGaussiansResult result;
        result.means = std::move(host_means);  // Move to avoid extra copy
        result.covs = std::move(host_covs);    // Move to avoid extra copy

        return result;
    }

    LocalGaussiansResult
    LocalGaussians(const std::vector<Vector<float, 3>> &points, const std::vector<unsigned int> &labels) {
        return {};
    }

    struct distance_calculator {
        __device__ __host__
        float operator()(const vec3 point, const vec3 object) const noexcept {
            return (point.x - object.x) * (point.x - object.x) +
                   (point.y - object.y) * (point.y - object.y) +
                   (point.z - object.z) * (point.z - object.z);
        }
    };

    __global__ void compute_means_and_covs_kernel(
            const dbvh d_bvh,
            const thrust::device_ptr<vec3> d_positions,
            thrust::device_ptr<vec3> d_means,
            thrust::device_ptr<mat3> d_covs,
            size_t num_objects, int k) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < num_objects) {
            const vec3 query = d_positions[idx];
            unsigned int indices[32];
            const auto num_found = query_device(d_bvh, knn(query, k),
                                                distance_calculator(), indices);

            vec3 mean = vec3::constant(0);
            mat3 cov = mat3::constant(0);

            // Combine mean and covariance calculation in one loop
            for (int i = 0; i < num_found; ++i) {
                const vec3 object = d_positions[indices[i]];
                mean = mean + object;
                cov = cov + outer(object, object);
            }

            mean = mean / num_found;
            cov = (cov - num_found * outer(mean, mean)) / num_found;


            mean = mean / num_found;
            cov = (cov - num_found * outer(mean, mean)) / num_found;

            // Write results to global memory
            d_means[idx] = mean;
            d_covs[idx] = cov;
        }
    }

    void ComputeCovs(const dbvh &d_bvh,
                     thrust::device_ptr<vec3> d_positions,
                     thrust::device_ptr<vec3> d_means,
                     thrust::device_ptr<mat3> d_covs,
                     size_t num_objects, int k) {
        int blockSize = 256; // Optimal block size based on experimentation
        int numBlocks = (num_objects + blockSize - 1) / blockSize;

        // Launch the kernel to compute means and covariances
        compute_means_and_covs_kernel<<<numBlocks, blockSize>>>(d_bvh, d_positions, d_means, d_covs, num_objects, k);

        // Check for errors without blocking
        cudaPeekAtLastError();
    }

    LocalGaussiansResult LocalGaussians(const std::vector<Vector<float, 3>> &points, int k) {
        size_t num_objects = points.size();
        thrust::host_vector<vec3> h_positions(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            h_positions[i] = vec3{points[i].x(), points[i].y(), points[i].z()};
        }
        LocalGaussiansDeviceData data = SetupLocalGaussiansDeviceData(h_positions);
        LocalGaussiansDevicePtr ptr = data.GetDevicePtr();

        ComputeCovs(ptr.d_bvh, ptr.d_positions, ptr.d_means, ptr.d_covs, num_objects, std::min(k, 32));
        return DeviceToHost(data);
    }
}