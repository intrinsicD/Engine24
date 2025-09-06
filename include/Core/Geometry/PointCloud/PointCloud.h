//
// Created by alex on 29.07.24.
//

#ifndef ENGINE24_POINTCLOUD_H
#define ENGINE24_POINTCLOUD_H

#include "PointCloudInterface.h"

namespace Bcg {
    class PointCloud {
    public:
        PointCloud() : interface(data) {
        }

        virtual ~PointCloud() = default;

        PointCloud(const PointCloud &other) : data(other.data),
                                              interface(data) {
        }

        PointCloud &operator=(const PointCloud &other) {
            if (this != &other) {
                data = other.data;
                interface = PointCloudInterface(data);
            }
            return *this;
        }

        PointCloud(PointCloud &&other) noexcept
            : data(std::move(other.data)),
              interface(other.data) {
        }

        PointCloud &operator=(PointCloud &&other) noexcept {
            if (this != &other) {
                data = std::move(other.data);
                interface = PointCloudInterface(data);
            }
            return *this;
        }

        PointCloudData data;
        PointCloudInterface interface;
    };
}

#endif //ENGINE24_POINTCLOUD_H
