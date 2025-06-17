//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_GPUMESH_H
#define ENGINE24_GPUMESH_H

#include <vector>
#include <cstdint>
#include "IAsset.h"

namespace Bcg {
    class Mesh : public IAsset {
    public:
        // Represents a single, contiguous part of the mesh that is drawn
        // with one material in a single draw call.
        struct SubMesh {
            uint32_t index_count;   // How many indices to draw for this sub-mesh
            uint32_t base_index;    // Offset into the master Index Buffer
            uint32_t base_vertex;   // Offset into the master Vertex Buffer
        };

        Mesh(uint32_t vao, uint32_t vbo, uint32_t ibo, uint32_t total_index_count, std::vector<SubMesh> &&sub_meshes);

        ~Mesh();

        // Deleted copy, enabled move constructors as before...

        uint32_t get_vao() const { return m_vao; }

        uint32_t get_total_index_count() const { return m_total_index_count; }

        const std::vector<SubMesh> &get_sub_meshes() const { return m_sub_meshes; }

    private:
        uint32_t m_vao = 0; // Vertex Array Object
        uint32_t m_vbo = 0; // Vertex Buffer Object
        uint32_t m_ibo = 0; // Index Buffer Object

        uint32_t m_total_index_count;
        std::vector<SubMesh> m_sub_meshes;
    };
}

#endif //ENGINE24_GPUMESH_H
