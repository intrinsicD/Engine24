//
// Created by alex on 18.06.24.
//

#include "PluginMesh.h"
#include "tiny_obj_loader.h"
#include "Logger.h"
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <sstream>
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "MeshCompute.h"
#include <iostream>
#include <chrono>
#include "Eigen/Core"
#include "pmp/surface_mesh.h"
#include "pmp/io/io.h"
#include "pmp/io/read_obj.h"
#include "pmp/io/read_off.h"
#include "pmp/io/read_stl.h"
#include "pmp/io/read_pmp.h"

namespace Bcg {
    pmp::SurfaceMesh PluginMesh::load(const char *path) {
        std::string ext = path;
        ext = ext.substr(ext.find_last_of('.') + 1);
        if (ext == "obj") {
            return load_obj(path);
        } else if (ext == "off") {
            return load_off(path);
        } else if (ext == "stl") {
            return load_stl(path);
        } else if (ext == "ply") {
            return load_ply(path);
        } else if (ext == "pmp") {
            return load_pmp(path);
        } else {
            Log::Error((std::string("Unsupported file format: ") + ext).c_str());
            return {};
        }
    }

    pmp::SurfaceMesh PluginMesh::load_obj(const char *path) {
        pmp::SurfaceMesh mesh;
        pmp::read_obj(mesh, path);
        return mesh;
    }

    pmp::SurfaceMesh PluginMesh::load_off(const char *path) {
        pmp::SurfaceMesh mesh;
        pmp::read_off(mesh, path);
        return mesh;
    }

    pmp::SurfaceMesh PluginMesh::load_stl(const char *path) {
        pmp::SurfaceMesh mesh;
        pmp::read_stl(mesh, path);
        merge_vertices(mesh, 0.0001f);
        return mesh;
    }

    pmp::SurfaceMesh PluginMesh::load_ply(const char *path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            Log::Error((std::string("Failed to open PLY file: ") + path).c_str());
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

        pmp::SurfaceMesh mesh;

        for (int i = 0; i < numVertices; ++i) {
            float x, y, z;
            file >> x >> y >> z;
            mesh.add_vertex(pmp::Point(x, y, z));
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
            mesh.add_triangle(pmp::Vertex(a), pmp::Vertex(b), pmp::Vertex(c));
        }

        file.close();
        return mesh;
    }

    pmp::SurfaceMesh PluginMesh::load_pmp(const char *path) {
        pmp::SurfaceMesh mesh;
        pmp::read_pmp(mesh, path);
        return mesh;
    }

    void PluginMesh::merge_vertices(pmp::SurfaceMesh &mesh, float tol) {
        struct VertexHash {
            size_t operator()(const pmp::Point &p) const {
                auto h1 = std::hash<float>{}(p[0]);
                auto h2 = std::hash<float>{}(p[1]);
                auto h3 = std::hash<float>{}(p[2]);
                return h1 ^ h2 ^ h3;
            }
        };

        struct VertexEqual {
            bool operator()(const pmp::Point &p1, const pmp::Point &p2) const {
                return pmp::norm(p1 - p2) < tol;
            }

            float tol;

            VertexEqual(float t) : tol(t) {}
        };

        std::unordered_map<pmp::Point, pmp::Vertex, VertexHash, VertexEqual> vertexMap(10, VertexHash(),
                                                                                       VertexEqual(tol));

        // Map to store the new vertex positions
        auto vertexReplacementMap = mesh.vertex_property<pmp::Vertex>("v:replacement");

        // Iterate over all vertices in the mesh
        for (auto v: mesh.vertices()) {
            pmp::Point p = mesh.position(v);

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
                pmp::Vertex to = mesh.to_vertex(h);
                mesh.set_vertex(h, vertexReplacementMap[to]);
            }
        }

        // Remove duplicate vertices
        std::vector<pmp::Vertex> vertices_to_delete;
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

            pmp::SurfaceMesh smesh = PluginMesh::load(event.paths[i]);
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
                auto mesh = PluginMesh::load(filePathName.c_str());
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void PluginMesh::render() {

    }
}