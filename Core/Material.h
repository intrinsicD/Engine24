//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATERIAL_H
#define ENGINE24_MATERIAL_H

#include <unordered_map>
#include <string>
#include "../MatVec.h"

namespace Bcg {
    struct Material {
        struct Attribute{
            std::string name;
            unsigned int location;
        };
        virtual ~Material() = default;

        unsigned int vao = -1;
        unsigned int program = -1;
        unsigned int offset = 0;
        unsigned int size = -1;

        std::vector<Attribute> attributes;

        std::unordered_map<std::string, unsigned int> textures;

        virtual void update_uniforms() = 0;
    };

    struct MeshMaterial : public Material {
        ~MeshMaterial() override = default;

        Vector<float, 3> base_color = Vector<float, 3>(1.0f, 1.0f, 1.0f);

        void update_uniforms() override{}
    };

    struct GraphMaterial : public Material {
        ~GraphMaterial() override = default;

        Vector<float, 3> base_color = Vector<float, 3>(1.0f, 1.0f, 1.0f);;

        void update_uniforms() override{}
    };

    struct PointCloudMaterial : public Material {
        ~PointCloudMaterial() override = default;

        Vector<float, 3> base_color = Vector<float, 3>(1.0f, 1.0f, 1.0f);;

        void update_uniforms() override{}
    };

}

#endif //ENGINE24_MATERIAL_H
