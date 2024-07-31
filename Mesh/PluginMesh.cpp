//
// Created by alex on 18.06.24.
//

#include "PluginMesh.h"
#include "Logger.h"
#include <fstream>
#include <unordered_map>
#include <sstream>
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "MeshGui.h"
#include <chrono>
#include "SurfaceMesh.h"
#include "io/io.h"
#include "io/read_obj.h"
#include "io/read_off.h"
#include "io/read_stl.h"
#include "io/read_pmp.h"
#include "Camera.h"
#include "GuiUtils.h"
#include "VertexArrayObject.h"
#include "Views.h"
#include "MeshCommands.h"
#include "EntityCommands.h"
#include "Picker.h"
#include "Transform.h"

namespace Bcg {
    namespace PluginMeshInternal{
        void on_drop_file(const Events::Callback::Drop &event) {
            PluginMesh plugin;
            for (int i = 0; i < event.count; ++i) {
                auto start_time = std::chrono::high_resolution_clock::now();

                SurfaceMesh smesh = PluginMesh::load(event.paths[i]);
                auto end_time = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> build_duration = end_time - start_time;
                Log::Info("Build Smesh in " + std::to_string(build_duration.count()) + " seconds");
            }
        }
    }

    SurfaceMesh PluginMesh::load(const std::string &path) {
        std::string ext = path;
        ext = ext.substr(ext.find_last_of('.') + 1);
        SurfaceMesh mesh;
        if (ext == "obj") {
            mesh = load_obj(path);
        } else if (ext == "off") {
            mesh = load_off(path);
        } else if (ext == "stl") {
            mesh = load_stl(path);
        } else if (ext == "ply") {
            mesh = load_ply(path);
        } else if (ext == "pmp") {
            mesh = load_pmp(path);
        } else {
            Log::Error("Unsupported file format: " + ext);
            return {};
        }
        auto entity_id = Engine::State().create();
        Commands::Entity::Add<SurfaceMesh>(entity_id, mesh, "Mesh").execute();
        Commands::Mesh::SetupForRendering(entity_id).execute();
        return mesh;
    }

    SurfaceMesh PluginMesh::load_obj(const std::string &path) {
        SurfaceMesh mesh;
        read_obj(mesh, path);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_off(const std::string &path) {
        SurfaceMesh mesh;
        read_off(mesh, path);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_stl(const std::string &path) {
        SurfaceMesh mesh;
        read_stl(mesh, path);
        merge_vertices(mesh, 0.0001f);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_ply(const std::string &path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            Log::Error("Failed to open PLY file: " + path);
            return {};
        }

        std::string line;
        std::string format;
        int numVertices = 0;
        int numFaces = 0;
        bool header = true;

        while (header && std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "format") {
                iss >> format;
            } else if (token == "element") {
                iss >> token;
                if (token == "vertex") {
                    iss >> numVertices;
                } else if (token == "face") {
                    iss >> numFaces;
                }
            } else if (token == "end_header") {
                header = false;
            }
        }

        SurfaceMesh mesh;

        for (int i = 0; i < numVertices; ++i) {
            float x, y, z;
            file >> x >> y >> z;
            mesh.add_vertex(Point(x, y, z));
        }

        for (int i = 0; i < numFaces; ++i) {
            int numVerticesInFace;
            file >> numVerticesInFace;
            if (numVerticesInFace != 3) {
                Log::Error("Only triangular faces are supported.");
                return {};
            }

            unsigned int a, b, c;
            file >> a >> b >> c;
            mesh.add_triangle(Vertex(a), Vertex(b), Vertex(c));
        }

        file.close();
        return mesh;
    }

    SurfaceMesh PluginMesh::load_pmp(const std::string &path) {
        SurfaceMesh mesh;
        read_pmp(mesh, path);
        return mesh;
    }

    void PluginMesh::merge_vertices(SurfaceMesh &mesh, float tol) {
        struct VertexHash {
            size_t operator()(const Point &p) const {
                auto h1 = std::hash<float>{}(p[0]);
                auto h2 = std::hash<float>{}(p[1]);
                auto h3 = std::hash<float>{}(p[2]);
                return h1 ^ h2 ^ h3;
            }
        };

        struct VertexEqual {
            bool operator()(const Point &p1, const Point &p2) const {
                return (p1 - p2).norm() < tol;
            }

            float tol;

            explicit VertexEqual(float t) : tol(t) {}
        };

        std::unordered_map<Point, Vertex, VertexHash, VertexEqual> vertexMap(10, VertexHash(),
                                                                             VertexEqual(tol));

        // Map to store the new vertex positions
        auto vertexReplacementMap = mesh.vertex_property<Vertex>("v:replacement");

        // Iterate over all vertices in the mesh
        for (auto v: mesh.vertices()) {
            Point p = mesh.position(v);

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

            auto h = mesh.halfedge(f);
            if (mesh.to_vertex(h) == mesh.to_vertex(mesh.next_halfedge(h)) ||
                mesh.to_vertex(h) == mesh.to_vertex(mesh.next_halfedge(mesh.next_halfedge(h))) ||
                mesh.to_vertex(mesh.next_halfedge(h)) == mesh.to_vertex(mesh.next_halfedge(mesh.next_halfedge(h)))) {
                mesh.delete_face(f);
            }
        }

        // Finalize the changes
        mesh.garbage_collection();
    }

    PluginMesh::PluginMesh() : Plugin("PluginMesh") {}

    void PluginMesh::activate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&PluginMeshInternal::on_drop_file>();
        Plugin::activate();
    }

    void PluginMesh::begin_frame() {

    }

    void PluginMesh::update() {

    }

    void PluginMesh::end_frame() {

    }

    void PluginMesh::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&PluginMeshInternal::on_drop_file>();
        Plugin::deactivate();
    }

    static bool show_mesh_gui = false;

    void PluginMesh::render_menu() {
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

    void PluginMesh::render_gui() {
        Gui::ShowLoadMesh();
        if (show_mesh_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("Mesh", &show_mesh_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowSurfaceMesh(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginMesh::render() {
        auto mesh_view = Engine::State().view<MeshView>();
        auto &camera = Engine::Context().get<Camera>();

        for (auto entity_id: mesh_view) {
            auto &mw = Engine::State().get<MeshView>(entity_id);

            mw.vao.bind();
            mw.program.use();
            mw.program.set_uniform3fv("lightPosition", camera.v_params.eye.data());

            if (Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                mw.program.set_uniform4fm("model", transform.data(), false);
            } else {
                mw.program.set_uniform4fm("model", Transform().data(), false);
            }

            mw.draw();
        }
    }
}