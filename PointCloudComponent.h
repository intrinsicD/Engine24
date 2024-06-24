//
// Created by alex on 24.06.24.
//

#ifndef ENGINE24_POINTCLOUDCOMPONENT_H
#define ENGINE24_POINTCLOUDCOMPONENT_H

#include <string>
#include <vector>
#include <unordered_map>

namespace Bcg{
    struct PointCloudComponent {
        unsigned int vao;
        unsigned int vbo;

        struct Material {
            std::string name;
            std::unordered_map<std::string, unsigned int> texids;
        };

        std::vector<float> vertices;
        std::vector<Material> materials;
    };
}

#endif //ENGINE24_POINTCLOUDCOMPONENT_H
