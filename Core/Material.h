//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_MATERIAL_H
#define ENGINE24_MATERIAL_H

#include <unordered_map>
#include <string>
#include "../MatVec.h"
#include "GuiUtils.h"

namespace Bcg {
    struct Material {
        struct Attribute {
            std::string name;
            unsigned int index;
            unsigned int size;
            unsigned int type;
            bool normalized;
            unsigned int stride;
            const void *pointer;
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

        void update_uniforms() override {}
    };

    struct GraphMaterial : public Material {
        ~GraphMaterial() override = default;

        Vector<float, 3> base_color = Vector<float, 3>(1.0f, 1.0f, 1.0f);;

        void update_uniforms() override {}
    };

    struct PointCloudMaterial : public Material {
        ~PointCloudMaterial() override = default;

        Vector<float, 3> base_color = Vector<float, 3>(1.0f, 1.0f, 1.0f);;

        void update_uniforms() override {}
    };

    namespace Gui {
        void Show(Material &material);

        void Show(Material::Attribute &attribute);

        void Show(MeshMaterial &material);

        void Show(GraphMaterial &material);

        void Show(PointCloudMaterial &material);
    }
}

#endif //ENGINE24_MATERIAL_H
