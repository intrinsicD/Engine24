//
// Created by alex on 24.06.24.
//

#ifndef ENGINE24_GRAPHCOMPONENT_H
#define ENGINE24_GRAPHCOMPONENT_H

#include <string>
#include <vector>
#include <unordered_map>

namespace Bcg{
    struct GraphComponent {
        unsigned int vao;
        unsigned int vbo;
        unsigned int ebo;

        struct Material {
            std::string name;
            std::unordered_map<std::string, unsigned int> texids;
        };

        std::vector<float> vertices;
        std::vector<unsigned int> indices;
        std::vector<Material> materials;
    };
}

#endif //ENGINE24_GRAPHCOMPONENT_H
