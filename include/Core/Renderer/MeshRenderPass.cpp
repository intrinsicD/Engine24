//
// Created by alex on 17.06.25.
//

#include "MeshRenderPass.h"
#include "Camera.h"
#include "RenderableMeshComponent.h"
#include "GpuMesh.h"
#include "WorldTransformComponent.h"
#include <glad/gl.h>
#include "entt/entity/registry.hpp"

namespace Bcg{
    void MeshRenderPass::execute(entt::registry& registry, const Camera& camera) {
        auto view = registry.view<RenderableMeshComponent, WorldTransformComponent>();

        for (auto entity : view) {
            const auto& renderable = registry.get<RenderableMeshComponent>(entity);
            if (!renderable.visible) continue;

            // --- Get Assets ---
            auto mesh = m_asset_manager.get_asset<Mesh>(renderable.mesh_handle);
            if (!mesh) continue;

            const auto& world_transform = registry.get<WorldTransformComponent>(entity);
            const glm::mat4& model_matrix = world_transform.world_transform;

            // --- The Multi-Material Loop ---
            for (size_t i = 0; i < mesh->get_sub_meshes().size(); ++i) {
                // Get the material for this specific sub-mesh
                AssetHandle material_handle = renderable.material_handles[i];
                auto material = m_asset_manager.get_asset<Material>(material_handle);
                if (!material) {
                    // Use a default error material if one isn't loaded
                    continue;
                }

                // *** HERE IS THE KEY INTERACTION ***

                // 1. Get the Shader Program from the Material
                auto shader = m_asset_manager.get_asset<Shader>(material->shader_handle);
                if (!shader) continue;

                // 2. The Render Pass Binds the Shader
                shader->Bind();

                // 3. The Render Pass Sets Engine-Level Uniforms (Camera, etc.)
                shader->SetUniformMat4("u_Model", model_matrix);

                // 4. The Render Pass Sets Material-Specific Uniforms
                shader->SetUniformVec4("u_Material.albedoColor", material->albedo_color);
                shader->SetUniformFloat("u_Material.metallic", material->metallic);
                shader->SetUniformFloat("u_Material.roughness", material->roughness);

                // 5. The Render Pass Binds Textures from the Material
                auto albedo_tex = m_asset_manager.get_asset<Texture>(material->albedo_texture_handle);
                if (albedo_tex) {
                    albedo_tex->Bind(0); // Bind to texture slot 0
                    shader->SetUniformInt("u_Material.albedoMap", 0);
                }
                // ... bind other textures to other slots ...

                // 6. The Render Pass issues the draw call
                const auto &sub_mesh = mesh->get_sub_meshes()[i];

                glBindVertexArray(mesh->get_vao());

                // 2. Issue the specific draw call.
                glDrawElementsBaseVertex(
                        GL_TRIANGLES,                                     // Mode: We are drawing triangles.
                        sub_mesh.index_count,                             // Count: The number of indices for this part.
                        GL_UNSIGNED_INT,                                  // Type: Our indices are 32-bit unsigned integers.
                        (void*)(sizeof(uint32_t) * sub_mesh.base_index),   // Indices: A byte offset into the IBO.
                        sub_mesh.base_vertex                              // Base Vertex: An offset to add to each vertex index.
                );

                // 3. Unbind the VAO (good practice to avoid accidental state changes).
                glBindVertexArray(0);
            }
        }
    }
}