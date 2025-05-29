//
// Created by alex on 29.05.25.
//

#ifndef ENGINE24_MAT_VEC_HELPER_CUH
#define ENGINE24_MAT_VEC_HELPER_CUH
// CudaHelpers.h
#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>
#include "mat_vec.cuh"    // your Bcg::vec2/3/4, mat2/3/4

namespace Bcg {
    namespace cuda_helpers {

        // ——— vec2 —————————————————————————————————————————————————————————————————

        inline thrust::device_vector<vec2>
        to_device_safe(const std::vector<glm::vec2> &h) {
            thrust::host_vector<vec2> hv(h.size());
            for (size_t i = 0; i < h.size(); ++i)
                hv[i] = vec2(h[i].x, h[i].y);
            return thrust::device_vector<vec2>(hv);
        }

        inline thrust::device_vector<vec2>
        to_device_blit(const std::vector<glm::vec2> &h) {
            static_assert(sizeof(glm::vec2) == sizeof(vec2),
                          "glm::vec2 and Bcg::vec2 must be bit-identical");
            thrust::device_vector<vec2> d(h.size());
            cudaMemcpy(
                    thrust::raw_pointer_cast(d.data()),
                    h.data(),
                    h.size() * sizeof(glm::vec2),
                    cudaMemcpyHostToDevice
            );
            return d;
        }

        // ——— vec3 —————————————————————————————————————————————————————————————————

        inline thrust::device_vector<vec3>
        to_device_safe(const std::vector<glm::vec3> &h) {
            thrust::host_vector<vec3> hv(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                auto &g = h[i];
                hv[i] = vec3(g.x, g.y, g.z);
            }
            return thrust::device_vector<vec3>(hv);
        }

        inline thrust::device_vector<vec3>
        to_device_blit(const std::vector<glm::vec3> &h) {
            static_assert(sizeof(glm::vec3) == sizeof(vec3),
                          "glm::vec3 and Bcg::vec3 must be bit-identical");
            thrust::device_vector<vec3> d(h.size());
            cudaMemcpy(
                    thrust::raw_pointer_cast(d.data()),
                    h.data(),
                    h.size() * sizeof(glm::vec3),
                    cudaMemcpyHostToDevice
            );
            return d;
        }


        // ——— vec4 —————————————————————————————————————————————————————————————————

        inline thrust::device_vector<vec4>
        to_device_safe(const std::vector<glm::vec4> &h) {
            thrust::host_vector<vec4> hv(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                auto &g = h[i];
                hv[i] = vec4(g.x, g.y, g.z, g.w);
            }
            return thrust::device_vector<vec4>(hv);
        }

        inline thrust::device_vector<vec4>
        to_device_blit(const std::vector<glm::vec4> &h) {
            static_assert(sizeof(glm::vec4) == sizeof(vec4),
                          "glm::vec4 and Bcg::vec4 must be bit-identical");
            thrust::device_vector<vec4> d(h.size());
            cudaMemcpy(
                    thrust::raw_pointer_cast(d.data()),
                    h.data(),
                    h.size() * sizeof(glm::vec4),
                    cudaMemcpyHostToDevice
            );
            return d;
        }


        // ——— mat2 —————————————————————————————————————————————————————————————————

        inline thrust::device_vector<mat2>
        to_device_safe(const std::vector<glm::mat2> &h) {
            thrust::host_vector<mat2> hv(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                auto &m = h[i];
                // glm::mat2 is column-major: m[c][r]
                hv[i] = mat2(
                        m[0][0], m[1][0],
                        m[0][1], m[1][1]
                );
            }
            return thrust::device_vector<mat2>(hv);
        }

        inline thrust::device_vector<mat2>
        to_device_blit(const std::vector<glm::mat2> &h) {
            static_assert(sizeof(glm::mat2) == sizeof(mat2),
                          "glm::mat2 and Bcg::mat2 must be bit-identical");
            thrust::device_vector<mat2> d(h.size());
            cudaMemcpy(
                    thrust::raw_pointer_cast(d.data()),
                    h.data(),
                    h.size() * sizeof(glm::mat2),
                    cudaMemcpyHostToDevice
            );
            return d;
        }


        // ——— mat3 —————————————————————————————————————————————————————————————————

        inline thrust::device_vector<mat3>
        to_device_safe(const std::vector<glm::mat3> &h) {
            thrust::host_vector<mat3> hv(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                auto &m = h[i];
                hv[i] = mat3(
                        // row 0
                        m[0][0], m[1][0], m[2][0],
                        // row 1
                        m[0][1], m[1][1], m[2][1],
                        // row 2
                        m[0][2], m[1][2], m[2][2]
                );
            }
            return thrust::device_vector<mat3>(hv);
        }

        inline thrust::device_vector<mat3>
        to_device_blit(const std::vector<glm::mat3> &h) {
            static_assert(sizeof(glm::mat3) == sizeof(mat3),
                          "glm::mat3 and Bcg::mat3 must be bit-identical");
            thrust::device_vector<mat3> d(h.size());
            cudaMemcpy(
                    thrust::raw_pointer_cast(d.data()),
                    h.data(),
                    h.size() * sizeof(glm::mat3),
                    cudaMemcpyHostToDevice
            );
            return d;
        }


        // ——— mat4 —————————————————————————————————————————————————————————————————

        inline thrust::device_vector<mat4>
        to_device_safe(const std::vector<glm::mat4> &h) {
            thrust::host_vector<mat4> hv(h.size());
            for (size_t i = 0; i < h.size(); ++i) {
                auto &m = h[i];
                hv[i] = mat4(
                        // row 0
                        m[0][0], m[1][0], m[2][0], m[3][0],
                        // row 1
                        m[0][1], m[1][1], m[2][1], m[3][1],
                        // row 2
                        m[0][2], m[1][2], m[2][2], m[3][2],
                        // row 3
                        m[0][3], m[1][3], m[2][3], m[3][3]
                );
            }
            return thrust::device_vector<mat4>(hv);
        }

        inline thrust::device_vector<mat4>
        to_device_blit(const std::vector<glm::mat4> &h) {
            static_assert(sizeof(glm::mat4) == sizeof(mat4),
                          "glm::mat4 and Bcg::mat4 must be bit-identical");
            thrust::device_vector<mat4> d(h.size());
            cudaMemcpy(
                    thrust::raw_pointer_cast(d.data()),
                    h.data(),
                    h.size() * sizeof(glm::mat4),
                    cudaMemcpyHostToDevice
            );
            return d;
        }

    }
} // namespace Bcg::cuda_helpers

// —————————————————————————————————————————————————————————————
// from device back to host
// —————————————————————————————————————————————————————————————

namespace Bcg { namespace cuda_helpers {

        // —— vec2 ——————————————————————————————————

        inline std::vector<glm::vec2>
        to_host_safe(const thrust::device_vector<vec2>& d)
        {
            thrust::host_vector<vec2> hv = d;                // copy D→H
            std::vector<glm::vec2> h(hv.size());
            for (size_t i = 0; i < hv.size(); ++i)
                h[i] = glm::vec2(hv[i][0], hv[i][1]);
            return h;
        }

        inline std::vector<glm::vec2>
        to_host_blit(const thrust::device_vector<vec2>& d)
        {
            static_assert(sizeof(glm::vec2) == sizeof(vec2),
                          "glm::vec2 and Bcg::vec2 must be bit-identical");
            std::vector<glm::vec2> h(d.size());
            cudaMemcpy(
                    h.data(),
                    thrust::raw_pointer_cast(d.data()),
                    d.size() * sizeof(vec2),
                    cudaMemcpyDeviceToHost
            );
            return h;
        }


        // —— vec3 ——————————————————————————————————

        inline std::vector<glm::vec3>
        to_host_safe(const thrust::device_vector<vec3>& d)
        {
            thrust::host_vector<vec3> hv = d;
            std::vector<glm::vec3> h(hv.size());
            for (size_t i = 0; i < hv.size(); ++i)
                h[i] = glm::vec3(hv[i][0], hv[i][1], hv[i][2]);
            return h;
        }

        inline std::vector<glm::vec3>
        to_host_blit(const thrust::device_vector<vec3>& d)
        {
            static_assert(sizeof(glm::vec3) == sizeof(vec3),
                          "glm::vec3 and Bcg::vec3 must be bit-identical");
            std::vector<glm::vec3> h(d.size());
            cudaMemcpy(
                    h.data(),
                    thrust::raw_pointer_cast(d.data()),
                    d.size() * sizeof(vec3),
                    cudaMemcpyDeviceToHost
            );
            return h;
        }


        // —— vec4 ——————————————————————————————————

        inline std::vector<glm::vec4>
        to_host_safe(const thrust::device_vector<vec4>& d)
        {
            thrust::host_vector<vec4> hv = d;
            std::vector<glm::vec4> h(hv.size());
            for (size_t i = 0; i < hv.size(); ++i)
                h[i] = glm::vec4(hv[i][0], hv[i][1], hv[i][2], hv[i][3]);
            return h;
        }

        inline std::vector<glm::vec4>
        to_host_blit(const thrust::device_vector<vec4>& d)
        {
            static_assert(sizeof(glm::vec4) == sizeof(vec4),
                          "glm::vec4 and Bcg::vec4 must be bit-identical");
            std::vector<glm::vec4> h(d.size());
            cudaMemcpy(
                    h.data(),
                    thrust::raw_pointer_cast(d.data()),
                    d.size() * sizeof(vec4),
                    cudaMemcpyDeviceToHost
            );
            return h;
        }


        // —— mat2 ——————————————————————————————————

        inline std::vector<glm::mat2>
        to_host_safe(const thrust::device_vector<mat2>& d)
        {
            thrust::host_vector<mat2> hv = d;
            std::vector<glm::mat2> h(hv.size());
            for (size_t i = 0; i < hv.size(); ++i) {
                auto &m = hv[i];
                // Bcg::mat2 stores columns in m.cols[c][r]
                glm::mat2 gm;
                gm[0][0] = m(0,0); gm[1][0] = m(0,1);
                gm[0][1] = m(1,0); gm[1][1] = m(1,1);
                h[i] = gm;
            }
            return h;
        }

        inline std::vector<glm::mat2>
        to_host_blit(const thrust::device_vector<mat2>& d)
        {
            static_assert(sizeof(glm::mat2) == sizeof(mat2),
                          "glm::mat2 and Bcg::mat2 must be bit-identical");
            std::vector<glm::mat2> h(d.size());
            cudaMemcpy(
                    h.data(),
                    thrust::raw_pointer_cast(d.data()),
                    d.size() * sizeof(mat2),
                    cudaMemcpyDeviceToHost
            );
            return h;
        }


        // —— mat3 ——————————————————————————————————

        inline std::vector<glm::mat3>
        to_host_safe(const thrust::device_vector<mat3>& d)
        {
            thrust::host_vector<mat3> hv = d;
            std::vector<glm::mat3> h(hv.size());
            for (size_t i = 0; i < hv.size(); ++i) {
                auto &m = hv[i];
                glm::mat3 gm;
                // column-major:
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        gm[c][r] = m(r,c);
                h[i] = gm;
            }
            return h;
        }

        inline std::vector<glm::mat3>
        to_host_blit(const thrust::device_vector<mat3>& d)
        {
            static_assert(sizeof(glm::mat3) == sizeof(mat3),
                          "glm::mat3 and Bcg::mat3 must be bit-identical");
            std::vector<glm::mat3> h(d.size());
            cudaMemcpy(
                    h.data(),
                    thrust::raw_pointer_cast(d.data()),
                    d.size() * sizeof(mat3),
                    cudaMemcpyDeviceToHost
            );
            return h;
        }


        // —— mat4 ——————————————————————————————————

        inline std::vector<glm::mat4>
        to_host_safe(const thrust::device_vector<mat4>& d)
        {
            thrust::host_vector<mat4> hv = d;
            std::vector<glm::mat4> h(hv.size());
            for (size_t i = 0; i < hv.size(); ++i) {
                auto &m = hv[i];
                glm::mat4 gm;
                for (int c = 0; c < 4; ++c)
                    for (int r = 0; r < 4; ++r)
                        gm[c][r] = m(r,c);
                h[i] = gm;
            }
            return h;
        }

        inline std::vector<glm::mat4>
        to_host_blit(const thrust::device_vector<mat4>& d)
        {
            static_assert(sizeof(glm::mat4) == sizeof(mat4),
                          "glm::mat4 and Bcg::mat4 must be bit-identical");
            std::vector<glm::mat4> h(d.size());
            cudaMemcpy(
                    h.data(),
                    thrust::raw_pointer_cast(d.data()),
                    d.size() * sizeof(mat4),
                    cudaMemcpyDeviceToHost
            );
            return h;
        }

    }} // namespace Bcg::cuda_helpers


#endif //ENGINE24_MAT_VEC_HELPER_CUH
