#pragma once

#include "PointCloudInterface.h"

namespace Bcg {
    void compute_vectorfields_of_gaussians(PointCloudInterface &pci) {
        auto scales = pci.vertices.vertex_property<Vector<float, 3> >("v:scale");
        auto rotations = pci.vertices.vertex_property<Vector<float, 4> >("v:rotation");
        auto local_frame_x = pci.vertices.vertex_property<Vector<float, 3> >("v:local_frame_x");
        auto local_frame_y = pci.vertices.vertex_property<Vector<float, 3> >("v:local_frame_y");
        auto local_frame_z = pci.vertices.vertex_property<Vector<float, 3> >("v:local_frame_z");

        for (const auto &v : pci.vertices) {
            const auto quat = rotations[v];
            const auto scale = scales[v];
            Matrix<float, 3, 3> R = glm::mat3_cast(glm::quat(quat.x, quat.y, quat.z, quat.w));
            local_frame_x[v] = R[0] * scale.x;
            local_frame_y[v] = R[1] * scale.y;
            local_frame_z[v] = R[2] * scale.z;
        }
    }
}