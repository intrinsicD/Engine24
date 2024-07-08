//
// Created by alex on 18.06.24.
//

#include "PluginMesh.h"
#include "Core/Logger.h"
#include <fstream>
#include <unordered_map>
#include <sstream>
#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "MeshCompute.h"
#include <chrono>
#include "SurfaceMesh/SurfaceMesh.h"
#include "SurfaceMesh/io/io.h"
#include "SurfaceMesh/io/read_obj.h"
#include "SurfaceMesh/io/read_off.h"
#include "SurfaceMesh/io/read_stl.h"
#include "SurfaceMesh/io/read_pmp.h"
#include "Mesh.h"
#include "Resources.h"
#include "GuiUtils.h"
#include "PropertiesGui.h"
#include "glad/gl.h"

namespace Bcg {
    struct MeshView {
        unsigned int vao, vbo, ebo;
        unsigned int program;
        unsigned int num_indices;
    };

    static MeshView setup(SurfaceMesh &mesh) {
        MeshView mw;
        mw.num_indices = mesh.n_faces() * 3;
        glGenVertexArrays(1, &mw.vao);
        glBindVertexArray(mw.vao);

        glGenBuffers(1, &mw.vbo);
        glBindBuffer(GL_ARRAY_BUFFER, mw.vbo);
        glBufferData(GL_ARRAY_BUFFER, mesh.n_vertices() * sizeof(Point), mesh.positions().data(), GL_STATIC_DRAW);

        auto triangles = extract_triangle_list(mesh);
        glGenBuffers(1, &mw.ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mw.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mw.num_indices * sizeof(unsigned int), triangles.data(), GL_STATIC_DRAW);

        mw.program = glCreateProgram();
        const char *vertex_shader_src = "#version 330 core\n"
                                        "layout (location = 0) in vec3 aPos;\n"
                                        "void main()\n"
                                        "{\n"
                                        "   gl_Position = vec4(aPos, 1.0);\n"
                                        "}\0";

        const char *fragment_shader_src = "#version 330 core\n"
                                          "out vec4 FragColor;\n"
                                          "void main()\n"
                                          "{\n"
                                          "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
                                          "}\n\0";

        unsigned int vertex_shader;
        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_src, nullptr);
        glCompileShader(vertex_shader);

        int success;
        char infoLog[512];
        glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertex_shader, 512, nullptr, infoLog);
            Log::Error("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + std::string(infoLog));
        }

        unsigned int fragment_shader;
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_src, nullptr);
        glCompileShader(fragment_shader);

        glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragment_shader, 512, nullptr, infoLog);
            Log::Error("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + std::string(infoLog));
        }

        glAttachShader(mw.program, vertex_shader);
        glAttachShader(mw.program, fragment_shader);

        glLinkProgram(mw.program);

        glGetProgramiv(mw.program, GL_LINK_STATUS, &success);

        if (!success) {
            glGetProgramInfoLog(mw.program, 512, nullptr, infoLog);
            Log::Error("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + std::string(infoLog));
        }

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
        glEnableVertexAttribArray(0);

        return mw;
    }

    SurfaceMesh PluginMesh::load(const char *path) {
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
            Log::Error((std::string("Unsupported file format: ") + ext).c_str());
            return {};
        }
        std::string message = "Loaded Mesh";
        message += " #v: " + std::to_string(mesh.n_vertices());
        message += " #e: " + std::to_string(mesh.n_edges());
        message += " #h: " + std::to_string(mesh.n_halfedges());
        message += " #f: " + std::to_string(mesh.n_faces());
        Log::Info(message);

        auto mw = setup(mesh);
        auto entity_id = Engine::State().create();
        Engine::State().emplace<MeshView>(entity_id, mw);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_obj(const char *path) {
        SurfaceMesh mesh;
        read_obj(mesh, path);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_off(const char *path) {
        SurfaceMesh mesh;
        read_off(mesh, path);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_stl(const char *path) {
        SurfaceMesh mesh;
        read_stl(mesh, path);
        merge_vertices(mesh, 0.0001f);
        return mesh;
    }

    SurfaceMesh PluginMesh::load_ply(const char *path) {
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

    SurfaceMesh PluginMesh::load_pmp(const char *path) {
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

            VertexEqual(float t) : tol(t) {}
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
            Resources<SurfaceMesh> meshes;
            meshes.create_from(smesh);
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

    void PluginMesh::render_gui(SurfaceMesh &mesh) {
        static std::pair<int, std::string> curr_property;
        Gui::Combo("Vertices", curr_property, mesh.vprops_);
    }

    void PluginMesh::render() {
        auto mesh_view = Engine::State().view<MeshView>();

        for (auto entity_id: mesh_view) {
            auto &mw = Engine::State().get<MeshView>(entity_id);

            glBindVertexArray(mw.vao);
            glUseProgram(mw.program);
            glDrawElements(GL_TRIANGLES, mw.num_indices, GL_UNSIGNED_INT, 0);
        }
    }
}