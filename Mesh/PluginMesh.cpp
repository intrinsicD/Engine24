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
#include "MeshCompute.h"
#include <chrono>
#include "SurfaceMesh.h"
#include "io/io.h"
#include "io/read_obj.h"
#include "io/read_off.h"
#include "io/read_stl.h"
#include "io/read_pmp.h"
#include "Mesh.h"
#include "Camera.h"
#include "GuiUtils.h"
#include "PropertiesGui.h"
#include "Picker.h"
#include "VertexArrayObject.h"
#include "Views.h"
#include "AABB.h"
#include "MeshCommands.h"

namespace Bcg {
    void PluginMesh::setup(SurfaceMesh &mesh, entt::entity entity_id) {
        if (entity_id == entt::null) {
            entity_id = Engine::State().create();
        }

        auto *mw = Engine::State().try_get<MeshView>(entity_id);

        if (!mw) {
            mw = &Engine::State().emplace<MeshView>(entity_id);
            mw->vao.create();
            mw->vbo.create();
            mw->ebo.create();
        }
    }

    static MeshView Setup(SurfaceMesh &mesh) {
        MeshView mw;
        mw.num_indices = mesh.n_faces() * 3;

        mw.vao.create();
        mw.vao.bind();

        auto v_normals = ComputeVertexNormals(mesh);
        size_t size_in_bytes_vertices = mesh.n_vertices() * sizeof(Point);
/*
        glGenBuffers(1, &mw.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, mw.vbo);

        glBufferData(GL_ARRAY_BUFFER, 2 * size_in_bytes_vertices, NULL, GL_STATIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, size_in_bytes_vertices, mesh.positions().data());


        glBufferSubData(GL_ARRAY_BUFFER, size_in_bytes_vertices, size_in_bytes_vertices, v_normals.data());
*/

        BufferLayout batched_buffer;

        auto &gpu_pos = batched_buffer.get_or_add(mesh.vpoint_.name().c_str());
        gpu_pos.size_in_bytes = size_in_bytes_vertices;
        gpu_pos.dims = 3;
        gpu_pos.size = sizeof(float);
        gpu_pos.normalized = false;
        gpu_pos.offset = 0;
        gpu_pos.data = mesh.positions().data();

        auto &gpu_v_normals = batched_buffer.get_or_add(v_normals.name().c_str());
        gpu_v_normals.size_in_bytes = size_in_bytes_vertices;
        gpu_v_normals.dims = 3;
        gpu_v_normals.size = sizeof(float);
        gpu_v_normals.normalized = false;
        gpu_v_normals.offset = size_in_bytes_vertices;
        gpu_v_normals.data = v_normals.data();

        mw.vbo.create();
        mw.vbo.bind();
        mw.vbo.buffer_data(nullptr, batched_buffer.total_size_bytes(), Buffer::Usage::STATIC_DRAW);
        for (const auto &[name, item]: batched_buffer.layout) {
            mw.vbo.buffer_sub_data(item.data, item.size_in_bytes, item.offset);
        }

        /*BatchedBuffer batched_buffer;
        batched_buffer.usage = GL_STATIC_DRAW;
        batched_buffer.target = GL_ARRAY_BUFFER;
        batched_buffer.type = GL_FLOAT;

        auto &gpu_pos = batched_buffer.get_or_add(mesh.vpoint_.name().c_str());
        gpu_pos.size_in_bytes = size_in_bytes_vertices;
        gpu_pos.dims = 3;
        gpu_pos.size = sizeof(float);
        gpu_pos.normalized = GL_FALSE;
        gpu_pos.offset = 0;
        gpu_pos.data = mesh.positions().data();

        auto &gpu_v_normals = batched_buffer.get_or_add(v_normals.name().c_str());
        gpu_v_normals.size_in_bytes = size_in_bytes_vertices;
        gpu_v_normals.dims = 3;
        gpu_v_normals.size = sizeof(float);
        gpu_v_normals.normalized = GL_FALSE;
        gpu_v_normals.offset = size_in_bytes_vertices;
        gpu_v_normals.data = v_normals.data();

        Graphics::setup_batched_buffer(batched_buffer);
        glBindBuffer(batched_buffer.target, batched_buffer.id);*/


        return mw;
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
        Engine::State().emplace_or_replace<SurfaceMesh>(entity_id, mesh);
        Commands::Mesh::SetupForRendering(entity_id).execute();

/*        std::string message = "Loaded Mesh";
        message += " #v: " + std::to_string(mesh.n_vertices());
        message += " #e: " + std::to_string(mesh.n_edges());
        message += " #h: " + std::to_string(mesh.n_halfedges());
        message += " #f: " + std::to_string(mesh.n_faces());
        Log::Info(message);


        auto mw = Setup(mesh);
        Engine::State().emplace<MeshView>(entity_id, mw);
        Engine::State().emplace<Transform>(entity_id, Transform::Identity());
        auto &aabb = Engine::State().emplace<AABB<float>>(entity_id);
        Build(aabb, mesh.positions());*/

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

    void on_drop_file(const Events::Callback::Drop &event) {
        PluginMesh plugin;
        for (int i = 0; i < event.count; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            SurfaceMesh smesh = PluginMesh::load(event.paths[i]);
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> build_duration = end_time - start_time;
            Log::Info("Build Smesh in " + std::to_string(build_duration.count()) + " seconds");

            start_time = std::chrono::high_resolution_clock::now();
            auto f_normals = ComputeFaceNormals(smesh);
            end_time = std::chrono::high_resolution_clock::now();
            build_duration = end_time - start_time;
            Log::Info("ComputeFaceNormals Smesh in " + std::to_string(build_duration.count()) + " seconds");
/*
            for(auto f : smesh.faces()){
                std::cout << f_normals[f] << std::endl;
            }*/

            start_time = std::chrono::high_resolution_clock::now();
            auto v_normals = ComputeVertexNormals(smesh);
            end_time = std::chrono::high_resolution_clock::now();
            build_duration = end_time - start_time;
            Log::Info("ComputeVertexNormals Smesh in " + std::to_string(build_duration.count()) + " seconds");
        }
    }

    PluginMesh::PluginMesh() : Plugin("PluginMesh") {}

    void PluginMesh::activate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&on_drop_file>();
        Plugin::activate();
    }

    void PluginMesh::begin_frame() {

    }

    void PluginMesh::update() {

    }

    void PluginMesh::end_frame() {

    }

    void PluginMesh::deactivate() {
        Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&on_drop_file>();
        Plugin::deactivate();
    }

    void PluginMesh::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::BeginMenu("Mesh")) {
                if (ImGui::MenuItem("Load Mesh")) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    config.path = "/home/alex/Dropbox/Work/Datasets";
                    ImGuiFileDialog::Instance()->OpenDialog("Load Mesh", "Choose File", ".obj,.off,.stl,.ply",
                                                            config);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginMesh::render_gui() {
        if (ImGuiFileDialog::Instance()->Display("Load Mesh", ImGuiWindowFlags_NoCollapse, ImVec2(200, 100))) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
                auto mesh = PluginMesh::load(filePathName);
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void PluginMesh::render_gui(SurfaceMesh &mesh) {
        static std::pair<int, std::string> curr_property;
        Gui::Combo("Vertices", curr_property, mesh.vprops_);
    }

    void PluginMesh::render() {
        auto mesh_view = Engine::State().view<MeshView>();
        auto &camera = Engine::Context().get<Camera>();
        auto lightDirection = (camera.v_params.center - camera.v_params.eye).normalized();

        for (auto entity_id: mesh_view) {
            auto &mw = Engine::State().get<MeshView>(entity_id);

            mw.vao.bind();
            mw.program.use();
            mw.program.set_uniform3fv("lightDir", lightDirection.data());

            mw.draw();
        }
    }
}