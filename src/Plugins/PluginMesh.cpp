//
// Created by alex on 18.06.24.
//

#include <unordered_map>
#include <chrono>

#include "PluginSurfaceMesh.h"
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "Entity.h"
#include "EventsCallbacks.h"
#include "EventsEntity.h"
#include "MeshGui.h"
#include "SurfaceMeshIo.h"
#include "VertexArrayObject.h"
#include "Picker.h"
#include "PluginViewSphere.h"
#include "SurfaceMeshCompute.h"
#include "PluginTransform.h"
#include "PluginHierarchy.h"
#include "PluginAABB.h"
#include "PluginCamera.h"
#include "PluginViewMesh.h"

namespace Bcg {

    static void on_drop_file(const Events::Callback::Drop &event) {
        PluginSurfaceMesh plugin;
        for (int i = 0; i < event.count; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            SurfaceMesh smesh = PluginSurfaceMesh::read(event.paths[i]);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> build_duration = end_time - start_time;
            Log::Info("Build Smesh in " + std::to_string(build_duration.count()) + " seconds");
        }
    }

    static void on_cleanup_components(const Events::Entity::CleanupComponents &event) {
        Commands::Cleanup<SurfaceMesh>(event.entity_id).execute();
    }

    SurfaceMesh PluginSurfaceMesh::read(const std::string &filepath) {
        std::string ext = filepath;
        ext = ext.substr(ext.find_last_of('.') + 1);
        SurfaceMesh mesh;
        if (!Read(filepath, mesh)) {
            Log::Error("Unsupported file format: " + ext);
            return {};
        }
        if (mesh.has_face_property("f:indices")) {

        }
        auto entity_id = Engine::State().create();
        Engine::State().emplace<SurfaceMesh>(entity_id, mesh);
        Commands::Setup<SurfaceMesh>(entity_id).execute();
        Commands::Setup<MeshView>(entity_id).execute();
        Commands::Setup<SphereView>(entity_id).execute();
        return mesh;
    }

    bool PluginSurfaceMesh::write(const std::string &filepath, const SurfaceMesh &mesh) {
        std::string ext = filepath;
        ext = ext.substr(ext.find_last_of('.') + 1);
        if (!Write(filepath, mesh)) {
            Log::Error("Unsupported file format: " + ext);
            return false;
        }
        return true;
    }

    void PluginSurfaceMesh::merge_vertices(SurfaceMesh &mesh, float tol) {
        struct VertexHash {
            size_t operator()(const PointType &p) const {
                auto h1 = std::hash<float>{}(p[0]);
                auto h2 = std::hash<float>{}(p[1]);
                auto h3 = std::hash<float>{}(p[2]);
                return h1 ^ h2 ^ h3;
            }
        };

        struct VertexEqual {
            bool operator()(const PointType &p1, const PointType &p2) const {
                return (p1 - p2).norm() < tol;
            }

            float tol;

            explicit VertexEqual(float t) : tol(t) {}
        };

        std::unordered_map<PointType, Vertex, VertexHash, VertexEqual> vertexMap(10, VertexHash(),
                                                                                 VertexEqual(tol));

        // Map to store the new vertex positions
        auto vertexReplacementMap = mesh.vertex_property<Vertex>("v:replacement");

        // Iterate over all vertices in the mesh
        for (auto v: mesh.vertices()) {
            PointType p = mesh.position(v);

            auto it = vertexMap.find(p);
            if (it == vertexMap.end()) {
                // Add unique vertex
                vertexMap[p] = v;
                vertexReplacementMap[v] = v;
            } else {
                // Update to point to the existing unique vertex
                vertexReplacementMap[v] = it->second;
            }
        }

        // Update the halfedges to use the new vertex indices
        for (auto v: mesh.vertices()) {
            if (vertexReplacementMap[v] != v) {
                continue;  // Skip already updated vertices
            }

            for (auto h: mesh.halfedges(v)) {
                Vertex to = mesh.to_vertex(h);
                mesh.set_vertex(h, vertexReplacementMap[to]);
            }
        }

        // Remove duplicate vertices
        std::vector<Vertex> vertices_to_delete;
        for (auto v: mesh.vertices()) {
            if (vertexReplacementMap[v] != v) {
                vertices_to_delete.push_back(v);
            }
        }

        for (auto v: vertices_to_delete) {
            mesh.delete_vertex(v);
        }

        // Remove degenerate faces
        for (auto f: mesh.faces()) {
            if (mesh.is_deleted(f)) {
                continue;
            }

            auto h = mesh.get_halfedge(f);
            if (mesh.to_vertex(h) == mesh.to_vertex(mesh.next_halfedge(h)) ||
                mesh.to_vertex(h) == mesh.to_vertex(mesh.next_halfedge(mesh.next_halfedge(h))) ||
                mesh.to_vertex(mesh.next_halfedge(h)) == mesh.to_vertex(mesh.next_halfedge(mesh.next_halfedge(h)))) {
                mesh.delete_face(f);
            }
        }

        // Finalize the changes
        mesh.garbage_collection();
    }

    PluginSurfaceMesh::PluginSurfaceMesh() : Plugin("PluginMesh") {}

    void PluginSurfaceMesh::activate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&on_drop_file>();
        Engine::Dispatcher().sink<Events::Entity::CleanupComponents>().connect<&on_cleanup_components>();
        Plugin::activate();
    }

    void PluginSurfaceMesh::begin_frame() {

    }

    void PluginSurfaceMesh::update() {

    }

    void PluginSurfaceMesh::end_frame() {

    }

    void PluginSurfaceMesh::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&on_drop_file>();
        Engine::Dispatcher().sink<Events::Entity::CleanupComponents>().disconnect<&on_cleanup_components>();
        Plugin::deactivate();
    }

    static bool show_mesh_gui = false;

    void PluginSurfaceMesh::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            if (ImGui::BeginMenu("Mesh")) {
                if (ImGui::MenuItem("Load Mesh")) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    config.path = "/home/alex/Dropbox/Work/Datasets";
                    ImGuiFileDialog::Instance()->OpenDialog("Load Mesh", "Choose File", ".obj,.off,.stl,.ply",
                                                            config);
                }
                if (ImGui::MenuItem("Instance", nullptr, &show_mesh_gui)) {

                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginSurfaceMesh::render_gui() {
        Gui::ShowLoadMesh();
        if (show_mesh_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("Mesh", &show_mesh_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowSurfaceMesh(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginSurfaceMesh::render() {

    }

    namespace Commands {
        void Load<SurfaceMesh>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            auto &mesh = Engine::require<SurfaceMesh>(entity_id);

            if (!Read(filepath, mesh)) {
                Log::Warn("Abort {} command", name);
                return;
            }

            if (!mesh.has_face_property("f:indices")) {
                Log::TODO("Implement: Mesh does not have faces, its a Point Cloud. Forward to Point Cloud stuff...");
            }
        }

        void Setup<SurfaceMesh>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            if (!Engine::has<SurfaceMesh>(entity_id)) {
                Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
            }

            auto &mesh = Engine::require<SurfaceMesh>(entity_id);

            Setup<AABB>(entity_id).execute();
            CenterAndScaleByAABB(entity_id, mesh.vpoint_.name()).execute();

            auto &aabb = Engine::require<AABB>(entity_id);
            auto &transform = *PluginTransform::setup(entity_id);
            auto &hierarchy = Engine::require<Hierarchy>(entity_id);

            std::string message = name + ": ";
            message += " #v: " + std::to_string(mesh.n_vertices());
            message += " #e: " + std::to_string(mesh.n_edges());
            message += " #h: " + std::to_string(mesh.n_halfedges());
            message += " #f: " + std::to_string(mesh.n_faces());
            message += " Done.";

            Log::Info(message);
            float d = aabb.diagonal().maxCoeff();
            CenterCameraAtDistance(aabb.center(), d).execute();
            ComputeSurfaceMeshVertexNormals(entity_id);
        }

        void Cleanup<SurfaceMesh>::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            if (!Engine::has<SurfaceMesh>(entity_id)) {
                Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
                return;
            }

            Engine::Dispatcher().trigger(Events::Entity::PreRemove<SurfaceMesh>{entity_id});
            Engine::State().remove<SurfaceMesh>(entity_id);
            Engine::Dispatcher().trigger(Events::Entity::PostRemove<SurfaceMesh>{entity_id});
            Log::Info("{} for entity {}", name, entity_id);
        }


        void ComputeFaceNormals::execute() const {
            if (!Engine::valid(entity_id)) {
                Log::Warn(name + "Entity is not valid. Abort Command");
                return;
            }

            if (!Engine::has<SurfaceMesh>(entity_id)) {
                Log::Warn(name + "Entity does not have a SurfaceMesh. Abort Command");
                return;
            }

            auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);

/*        auto v_normals = ComputeFaceNormals(entity_id, mesh);*/
        }
    }
}