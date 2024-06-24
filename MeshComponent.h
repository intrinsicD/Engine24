//
// Created by alex on 24.06.24.
//

#ifndef ENGINE24_MESHCOMPONENT_H
#define ENGINE24_MESHCOMPONENT_H

#include <string>
#include <vector>
#include <unordered_map>

namespace Bcg {
    struct MeshComponent {
        unsigned int vao;
        unsigned int vbo;
        unsigned int ebo;

        struct Material {
            std::string name;
            std::unordered_map<std::string, unsigned int> texture_ids;
        };

        std::vector<float> positions;
        std::vector<unsigned int> triangles;
        std::vector<Material> materials;
    };
}

#endif //ENGINE24_MESHCOMPONENT_H
