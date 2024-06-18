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

namespace Bcg {
    MeshComponent PluginMesh::load(const char *path) {
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
        } else {
            Log::Error((std::string("Unsupported file format: ") + ext).c_str());
            return {};
        }
    }

    MeshComponent PluginMesh::load_obj(const char *path) {
        struct Vertex {
            float position[3];
            float normal[3];
            float texcoord[2];
        };

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path);
        if (!warn.empty()) {
            Log::Warn(warn.c_str());
        }
        if (!err.empty()) {
            Log::Error(err.c_str());
        }
        if (!ret) {
            return {};
        }

        MeshComponent mesh;
        for (const auto &shape: shapes) {
            for (const auto &index: shape.mesh.indices) {
                mesh.vertices.push_back(attrib.vertices[3 * index.vertex_index + 0]);
                mesh.vertices.push_back(attrib.vertices[3 * index.vertex_index + 1]);
                mesh.vertices.push_back(attrib.vertices[3 * index.vertex_index + 2]);

                if (!attrib.normals.empty()) {
                    mesh.vertices.push_back(attrib.normals[3 * index.normal_index + 0]);
                    mesh.vertices.push_back(attrib.normals[3 * index.normal_index + 1]);
                    mesh.vertices.push_back(attrib.normals[3 * index.normal_index + 2]);
                }

                if (!attrib.texcoords.empty()) {
                    mesh.vertices.push_back(attrib.texcoords[2 * index.texcoord_index + 0]);
                    mesh.vertices.push_back(attrib.texcoords[2 * index.texcoord_index + 1]);
                }

                mesh.indices.push_back(mesh.indices.size());
            }
        }
        return std::move(mesh);
    }

    MeshComponent PluginMesh::load_off(const char *path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            Log::Error((std::string("Failed to open OFF file: ") + path).c_str());
            return {};
        }

        std::string line;
        std::getline(file, line); // OFF header

        if (line != "OFF") {
            Log::Error((std::string("Not a valid OFF file: ") + path).c_str());
            return {};
        }

        int numVertices, numFaces, numEdges;
        file >> numVertices >> numFaces >> numEdges;

        MeshComponent mesh;

        mesh.vertices.reserve(numVertices * 3);
        mesh.indices.reserve(numFaces * 3);

        for (int i = 0; i < numVertices; ++i) {
            float x, y, z;
            file >> x >> y >> z;
            mesh.vertices.push_back(x);
            mesh.vertices.push_back(y);
            mesh.vertices.push_back(z);
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
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);
        }

        file.close();
        return std::move(mesh);
    }

    MeshComponent PluginMesh::load_stl(const char *path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            Log::Error((std::string("Failed to open STL file: ") + path).c_str());
            return {};
        }

        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(fileSize);
        file.read(buffer.data(), fileSize);

        if (buffer.size() < 84) {
            Log::Error((std::string("Invalid STL file: ") + path).c_str());
        }

        MeshComponent mesh;

        unsigned int numTriangles = *reinterpret_cast<unsigned int *>(&buffer[80]);
        mesh.vertices.reserve(numTriangles * 9);
        mesh.indices.reserve(numTriangles * 3);

        for (unsigned int i = 0; i < numTriangles; ++i) {
            unsigned int offset = 84 + i * 50;
            for (unsigned int j = 0; j < 3; ++j) {
                float x = *reinterpret_cast<float *>(&buffer[offset + 12 + j * 12]);
                float y = *reinterpret_cast<float *>(&buffer[offset + 16 + j * 12]);
                float z = *reinterpret_cast<float *>(&buffer[offset + 20 + j * 12]);
                mesh.vertices.push_back(x);
                mesh.vertices.push_back(y);
                mesh.vertices.push_back(z);
            }
            mesh.indices.push_back(i * 3);
            mesh.indices.push_back(i * 3 + 1);
            mesh.indices.push_back(i * 3 + 2);
        }

        file.close();
        merge_vertices(mesh, 0.0001f);
        return mesh;
    }

    MeshComponent PluginMesh::load_ply(const char *path) {
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

        MeshComponent mesh;

        mesh.vertices.reserve(numVertices * 3);
        mesh.indices.reserve(numFaces * 3);

        for (int i = 0; i < numVertices; ++i) {
            float x, y, z;
            file >> x >> y >> z;
            mesh.vertices.push_back(x);
            mesh.vertices.push_back(y);
            mesh.vertices.push_back(z);
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
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);
        }

        file.close();
        return mesh;
    }

    void PluginMesh::merge_vertices(MeshComponent &mesh, float tol) {
        std::unordered_map<size_t, unsigned int> vertexMap;
        std::vector<float> newVertices;
        std::vector<unsigned int> newIndices;
        size_t stride = 3;

        auto hashVertex = [tol](float x, float y, float z) {
            auto h1 = std::hash<float>{}(std::floor(x / tol));
            auto h2 = std::hash<float>{}(std::floor(y / tol));
            auto h3 = std::hash<float>{}(std::floor(z / tol));
            return h1 ^ h2 ^ h3;
        };

        for (size_t i = 0; i < mesh.vertices.size(); i += stride) {
            float x = mesh.vertices[i];
            float y = mesh.vertices[i + 1];
            float z = mesh.vertices[i + 2];
            size_t vertexHash = hashVertex(x, y, z);

            if (vertexMap.find(vertexHash) == vertexMap.end()) {
                vertexMap[vertexHash] = newVertices.size() / stride;
                newVertices.push_back(x);
                newVertices.push_back(y);
                newVertices.push_back(z);
            }
            newIndices.push_back(vertexMap[vertexHash]);
        }

        mesh.vertices = newVertices;
        mesh.indices = newIndices;
    }

    void PluginMesh::activate() {

    }

    void PluginMesh::update() {

    }

    void PluginMesh::deactivate() {

    }

    void PluginMesh::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            if (ImGui::BeginMenu("Mesh")) {
                if (ImGui::MenuItem("Load Mesh")) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".obj,.off,.stl, .ply", config);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void PluginMesh::render_gui() {
        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
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