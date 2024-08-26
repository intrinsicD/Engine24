//
// Created by alex on 13.08.24.
//

#include "MeshInterface.h"
#include "SurfaceMeshIo.h"
#include "SurfaceMeshTriangles.h"
#include "IoHelpers.h"
#include "Logger.h"
#include "happly.h"
#include <filesystem>
#include <map>

namespace Bcg {
    bool Read(const std::string &filepath, SurfaceMesh &mesh) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".obj") {
            return ReadObj(filepath, mesh);
        } else if (ext == ".off") {
            return ReadOff(filepath, mesh);
        } else if (ext == ".ply") {
            return ReadPly(filepath, mesh);
        } else if (ext == ".stl") {
            return ReadStl(filepath, mesh);
        } else if (ext == ".pmp") {
            return ReadPmp(filepath, mesh);
        } else {
            return false;
        }
    }

    bool ReadObj(const std::string &filepath, SurfaceMesh &mesh) {
        std::array<char, 200> s;
        float x, y, z;
        std::vector<Vertex> vertices;
        std::vector<TexCoordType> all_tex_coords; //individual texture coordinates
        std::vector<int>
                halfedge_tex_idx; //texture coordinates sorted for halfedges
        HalfedgeProperty<TexCoordType> tex_coords =
                mesh.halfedge_property<TexCoordType>("h:tex");
        bool with_tex_coord = false;

        // open file (in ASCII mode)
        FILE *in = fopen(filepath.c_str(), "r");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // clear line once
        memset(s.data(), 0, 200);

        // parse line by line (currently only supports vertex positions & faces
        while (in && !feof(in) && fgets(s.data(), 200, in)) {
            // comment
            if (s[0] == '#' || isspace(s[0]))
                continue;

                // vertex
            else if (strncmp(s.data(), "v ", 2) == 0) {
                if (sscanf(s.data(), "v %f %f %f", &x, &y, &z)) {
                    mesh.add_vertex(PointType(x, y, z));
                }
            }

                // normal
            else if (strncmp(s.data(), "vn ", 3) == 0) {
                if (sscanf(s.data(), "vn %f %f %f", &x, &y, &z)) {
                    // problematic as it can be either a vertex property when interpolated
                    // or a halfedge property for hard edges
                }
            }

                // texture coordinate
            else if (strncmp(s.data(), "vt ", 3) == 0) {
                if (sscanf(s.data(), "vt %f %f", &x, &y)) {
                    all_tex_coords.emplace_back(x, y);
                }
            }

                // face
            else if (strncmp(s.data(), "f ", 2) == 0) {
                int component(0);
                bool end_of_vertex(false);
                char *p0, *p1(s.data() + 1);

                vertices.clear();
                halfedge_tex_idx.clear();

                // skip white-spaces
                while (*p1 == ' ')
                    ++p1;

                while (p1) {
                    p0 = p1;

                    // overwrite next separator

                    // skip '/', '\n', ' ', '\0', '\r' <-- don't forget Windows
                    while (*p1 != '/' && *p1 != '\r' && *p1 != '\n' && *p1 != ' ' &&
                           *p1 != '\0')
                        ++p1;

                    // detect end of vertex
                    if (*p1 != '/') {
                        end_of_vertex = true;
                    }

                    // replace separator by '\0'
                    if (*p1 != '\0') {
                        *p1 = '\0';
                        p1++; // point to next token
                    }

                    // detect end of line and break
                    if (*p1 == '\0' || *p1 == '\n') {
                        p1 = nullptr;
                    }

                    // read next vertex component
                    if (*p0 != '\0') {
                        switch (component) {
                            case 0: // vertex
                            {
                                int idx = atoi(p0);
                                if (idx < 0)
                                    idx = mesh.n_vertices() + idx + 1;
                                vertices.emplace_back(idx - 1);
                                break;
                            }
                            case 1: // texture coord
                            {
                                int idx = atoi(p0) - 1;
                                halfedge_tex_idx.push_back(idx);
                                with_tex_coord = true;
                                break;
                            }
                            case 2: // normal
                                break;
                        }
                    }

                    ++component;

                    if (end_of_vertex) {
                        component = 0;
                        end_of_vertex = false;
                    }
                }

                Face f;
                try {
                    f = mesh.add_face(vertices);
                } catch (const TopologyException &e) {
                    Log::Warn("Failed to add face: {}", e.what());
                }

                // add texture coordinates
                if (with_tex_coord && f.is_valid()) {
                    auto h_fit = mesh.halfedges(f);
                    auto h_end = h_fit;
                    unsigned v_idx = 0;
                    do {
                        tex_coords[*h_fit] =
                                all_tex_coords.at(halfedge_tex_idx.at(v_idx));
                        ++v_idx;
                        ++h_fit;
                    } while (h_fit != h_end);
                }
            }
            // clear line
            memset(s.data(), 0, 200);
        }

        // if there are no textures, delete texture property!
        if (!with_tex_coord) {
            mesh.remove_halfedge_property(tex_coords);
        }

        fclose(in);
        return true;
    }


    void ReadOffAscii(SurfaceMesh &mesh, FILE *in, const bool has_normals,
                      const bool has_texcoords, const bool has_colors,
                      char *first_line) {
        std::array<char, 1000> line;
        char *lp = first_line;
        int nc;
        long int i, j, idx;
        long int nv, nf, ne;
        float x, y, z, r, g, b;
        Vertex v;

        // properties
        VertexProperty<NormalType> normals;
        VertexProperty<TexCoordType> texcoords;
        VertexProperty<ColorType> colors;
        if (has_normals)
            normals = mesh.vertex_property<NormalType>("v:normal");
        if (has_texcoords)
            texcoords = mesh.vertex_property<TexCoordType>("v:tex");
        if (has_colors)
            colors = mesh.vertex_property<ColorType>("v:color");

        // read line, but skip comment lines
        while (lp && (lp[0] == '#' || lp[0] == '\n' || lp[0] == '\r')) {
            lp = fgets(line.data(), 1000, in);
        }

        // #Vertices, #Faces, #Edges
        auto items = sscanf(lp, "%ld %ld %ld\n", &nv, &nf, &ne);

        if (items < 3 || ne < 0) {
            Log::Error("Failed to parse OFF header");
            return;
        }

        if (nv < 1) {
            Log::Error("Off file has no vertices");
            return;
        }

        mesh.reserve(nv, std::max(3 * nv, ne), nf);

        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nv && !feof(in); ++i) {
            // read line, but skip comment lines
            do {
                lp = fgets(line.data(), 1000, in);
            } while (lp && (lp[0] == '#' || lp[0] == '\n'));
            lp = line.data();

            // position
            items = sscanf(lp, "%f %f %f%n", &x, &y, &z, &nc);
            assert(items == 3);
            v = mesh.add_vertex(PointType(x, y, z));
            lp += nc;

            // normal
            if (has_normals) {
                if (sscanf(lp, "%f %f %f%n", &x, &y, &z, &nc) == 3) {
                    normals[v] = NormalType(x, y, z);
                }
                lp += nc;
            }

            // color
            if (has_colors) {
                if (sscanf(lp, "%f %f %f%n", &r, &g, &b, &nc) == 3) {
                    if (r > 1.0f || g > 1.0f || b > 1.0f) {
                        r /= 255.0f;
                        g /= 255.0f;
                        b /= 255.0f;
                    }
                    colors[v] = ColorType(r, g, b);
                }
                lp += nc;
            }

            // tex coord
            if (has_texcoords) {
                items = sscanf(lp, "%f %f%n", &x, &y, &nc);
                assert(items == 2);
                texcoords[v][0] = x;
                texcoords[v][1] = y;
                lp += nc;
            }
        }

        if (nf < 1) {
            Log::Warn("Off file has no faces");
            return;
        }

        // read faces: #N v[1] v[2] ... v[n-1]
        std::vector<Vertex> vertices;
        for (i = 0; i < nf; ++i) {
            // read line, but skip comment lines
            do {
                lp = fgets(line.data(), 1000, in);
            } while (lp && (lp[0] == '#' || lp[0] == '\n'));
            lp = line.data();

            // #vertices
            items = sscanf(lp, "%ld%n", &nv, &nc);
            assert(items == 1);
            if (nv < 1)
                throw IOException("Invalid index count");
            vertices.resize(nv);
            lp += nc;

            // indices
            for (j = 0; j < nv; ++j) {
                items = sscanf(lp, "%ld%n", &idx, &nc);
                assert(items == 1);
                if (idx < 0)
                    throw IOException("Invalid index");
                vertices[j] = Vertex(idx);
                lp += nc;
            }
            try {
                mesh.add_face(vertices);
            }
            catch (const TopologyException &e) {
                Log::Warn("Failed to add face: {}", e.what());
            }
        }
    }

    template<typename T>
    requires(sizeof(T) == 4)
    void ReadBinary(FILE *in, T &t, bool swap = false) {
        [[maybe_unused]] auto n_items = fread((char *) &t, 1, sizeof(t), in);

        if (swap) {
            const auto u32v = std::bit_cast<uint32_t>(t);
            const auto vv = Byteswap32(u32v);
            t = std::bit_cast<T>(vv);
        }
    }

    void ReadOffBinary(SurfaceMesh &mesh, FILE *in, const bool has_normals,
                       const bool has_texcoords, const bool has_colors,
                       const std::filesystem::path &file) {
        uint32_t i, j, idx(0);
        uint32_t nv(0), nf(0), ne(0);
        PointType p, n;
        TexCoordType t;
        Vertex v;

        // binary cannot (yet) read colors
        if (has_colors)
            throw IOException("Colors not supported for binary OFF file.");

        // properties
        VertexProperty<NormalType> normals;
        VertexProperty<TexCoordType> texcoords;
        if (has_normals)
            normals = mesh.vertex_property<NormalType>("v:normal");
        if (has_texcoords)
            texcoords = mesh.vertex_property<TexCoordType>("v:tex");

        // #Vertices, #Faces, #Edges
        ReadBinary(in, nv);

        // Check for little endian encoding used by previous versions.
        // Swap the ordering if the total file size is smaller than the size
        // required to store all vertex coordinates.
        auto file_size = std::filesystem::file_size(file);
        bool swap = file_size < nv * 3 * 4 ? true : false;
        if (swap)
            nv = Byteswap32(nv);

        ReadBinary(in, nf, swap);
        ReadBinary(in, ne, swap);
        mesh.reserve(nv, std::max(3 * nv, ne), nf);

        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nv && !feof(in); ++i) {
            // position
            ReadBinary(in, p[0], swap);
            ReadBinary(in, p[1], swap);
            ReadBinary(in, p[2], swap);
            v = mesh.add_vertex((PointType) p);

            // normal
            if (has_normals) {
                ReadBinary(in, n[0], swap);
                ReadBinary(in, n[1], swap);
                ReadBinary(in, n[2], swap);
                normals[v] = (NormalType) n;
            }

            // tex coord
            if (has_texcoords) {
                ReadBinary(in, t[0], swap);
                ReadBinary(in, t[1], swap);
                texcoords[v][0] = t[0];
                texcoords[v][1] = t[1];
            }
        }

        // read faces: #N v[1] v[2] ... v[n-1]
        std::vector<Vertex> vertices;
        for (i = 0; i < nf; ++i) {
            ReadBinary(in, nv, swap);
            vertices.resize(nv);
            for (j = 0; j < nv; ++j) {
                ReadBinary(in, idx, swap);
                vertices[j] = Vertex(idx);
            }
            try {
                mesh.add_face(vertices);
            }
            catch (const TopologyException &e) {
                Log::Warn("Failed to add face: {}", e.what());
            }
        }
    }

    bool ReadOff(const std::string &filepath, SurfaceMesh &mesh) {
        std::array<char, 200> line;
        bool has_texcoords = false;
        bool has_normals = false;
        bool has_colors = false;
        bool has_hcoords = false;
        bool has_dim = false;
        bool is_binary = false;

        // open file (in ASCII mode)
        FILE *in = fopen(filepath.c_str(), "r");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // read header: [ST][C][N][4][n]OFF BINARY
        char *c = fgets(line.data(), 200, in);
        assert(c != nullptr);
        c = line.data();
        if (c[0] == 'S' && c[1] == 'T') {
            has_texcoords = true;
            c += 2;
        }
        if (c[0] == 'C') {
            has_colors = true;
            ++c;
        }
        if (c[0] == 'N') {
            has_normals = true;
            ++c;
        }
        if (c[0] == '4') {
            has_hcoords = true;
            ++c;
        }
        if (c[0] == 'n') {
            has_dim = true;
            ++c;
        }
        if (strncmp(c, "OFF", 3) != 0) {
            fclose(in);
            Log::Error("Failed to parse OFF header");
            return false;
        }
        c += 3;
        if (c[0] == ' ')
            ++c;
        if (strncmp(c, "BINARY", 6) == 0) {
            is_binary = true;
            c += 6;
        }
        if (c[0] == ' ')
            ++c;

        if (has_hcoords) {
            fclose(in);
            Log::Error("Error: Homogeneous coordinates not supported.");
            return false;
        }
        if (has_dim) {
            fclose(in);
            Log::Error("Error: vertex dimension != 3 not supported");
            return false;
        }

        // if binary: reopen file in binary mode
        if (is_binary) {
            fclose(in);
            in = fopen(filepath.c_str(), "rb");
            c = fgets(line.data(), 200, in);
            assert(c != nullptr);
        }

        // read as ASCII or binary
        if (is_binary) {
            ReadOffBinary(mesh, in, has_normals, has_texcoords, has_colors, filepath);
        } else {
            ReadOffAscii(mesh, in, has_normals, has_texcoords, has_colors, c);
        }

        fclose(in);
        return true;;
    }

    bool ReadPly(const std::string &filepath, SurfaceMesh &mesh) {
        happly::PLYData plyIn(filepath);

        if (!plyIn.hasElement("vertex") ||
            !plyIn.getElement("vertex").hasProperty("x") ||
            !plyIn.getElement("vertex").hasProperty("y") ||
            !plyIn.getElement("vertex").hasProperty("z")) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
        std::vector<std::array<unsigned char, 3>> vCol;
        if (plyIn.getElement("vertex").hasProperty("red")) {
            vCol = plyIn.getVertexColors();
        }
        std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();

        auto colors = mesh.vertex_property<Vector<float, 3>>("v:color");

        mesh.vprops_.reserve(vPos.size());
        for (const auto &point: vPos) {
            mesh.add_vertex(Vector<double, 3>(point.data()).cast<float>());
        }

        if (!vCol.empty()) {
            for (const auto &v: mesh.vertices()) {
                const Eigen::Vector<double, 3> color(vCol[v.idx()][0] / 255.0, vCol[v.idx()][1] / 255.0,
                                                     vCol[v.idx()][2] / 255.0);
                colors[v] = color.cast<float>();
            }
        } else {
            mesh.vprops_.remove(colors);
        }

        mesh.fprops_.reserve(fInd.size());
        for (const auto &face: fInd) {
            mesh.add_face({Vertex(face[0]), Vertex(face[1]), Vertex(face[2])});
        }

        return mesh.fprops_.size() > 0;
    }

    using vec3 = Vector<float, 3>;

    struct CompareVec3 {
        bool operator()(const vec3 &v0, const vec3 &v1) const {
            if (fabs(v0[0] - v1[0]) <= eps_) {
                if (fabs(v0[1] - v1[1]) <= eps_) {
                    return (v0[2] < v1[2] - eps_);
                } else
                    return (v0[1] < v1[1] - eps_);
            } else
                return (v0[0] < v1[0] - eps_);
        }

        ScalarType eps_{std::numeric_limits<ScalarType>::min()};
    };


    bool ReadStl(const std::string &filepath, SurfaceMesh &mesh) {
        std::array<char, 100> line;
        uint32_t i, nT(0);
        vec3 p;
        Vertex v;
        std::vector<Vertex> vertices(3);

        CompareVec3 comp;
        std::map<vec3, Vertex, CompareVec3> vertex_map(comp);

        // open file (in ASCII mode)
        FILE *in = fopen(filepath.c_str(), "r");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // determine if the file is a binary STL file
        auto is_binary = [&]() {
            [[maybe_unused]] auto c = fgets(line.data(), 6, in);

            // if the file does *not* start with "solid" we have a binary file
            if ((strncmp(line.data(), "SOLID", 5) != 0) &&
                (strncmp(line.data(), "solid", 5) != 0)) {
                return true;
            }

            // otherwise check if file size matches number of triangles
            auto fp = fopen(filepath.c_str(), "rb");
            if (!fp) {
                Log::Error("Failed to open file: " + filepath);
                return false;
            }

            // skip header
            [[maybe_unused]] auto n_items = fread(line.data(), 1, 80, fp);

            // read number of triangles
            uint32_t n_triangles{0};
            TFRead(fp, n_triangles);

            // get file size minus header and element count
            fseek(fp, 0L, SEEK_END);
            auto size = ftell(fp);
            size -= 84;
            fclose(fp);

            // for each triangle we should have 4*12+2 bytes:
            // normal, x,y,z, attribute byte count
            auto predicted = (4 * 12 + 2) * n_triangles;

            return size == predicted;
        };

        // parse binary STL
        if (is_binary()) {
            // re-open file in binary mode
            fclose(in);
            in = fopen(filepath.c_str(), "rb");
            if (!in) {
                Log::Error("Failed to open file: " + filepath);
                return false;
            }

            // skip dummy header
            [[maybe_unused]] auto n_items = fread(line.data(), 1, 80, in);
            assert(n_items > 0);

            // read number of triangles
            TFRead(in, nT);

            // read triangles
            while (nT) {
                // skip triangle normal
                n_items = fread(line.data(), 1, 12, in);
                assert(n_items > 0);

                // triangle's vertices
                for (i = 0; i < 3; ++i) {
                    TFRead(in, p);

                    // has vector been referenced before?
                    auto it = vertex_map.find(p);
                    if (it == vertex_map.end()) {
                        // No : add vertex and remember idx/vector mapping
                        v = mesh.add_vertex((PointType) p);
                        vertices[i] = v;
                        vertex_map[p] = v;
                    } else {
                        // Yes : get index from map
                        vertices[i] = it->second;
                    }
                }

                // Add face only if it is not degenerated
                if ((vertices[0] != vertices[1]) && (vertices[0] != vertices[2]) &&
                    (vertices[1] != vertices[2])) {
                    try {
                        mesh.add_face(vertices);
                    }
                    catch (const TopologyException &e) {
                        Log::Warn("Failed to add face: {}", e.what());
                    }
                }

                n_items = fread(line.data(), 1, 2, in);
                assert(n_items > 0);

                --nT;
            }
        }

            // parse ASCII STL
        else {
            char *c{nullptr};

            // parse line by line
            while (in && !feof(in) && fgets(line.data(), 100, in)) {
                // skip white-space
                for (c = line.data(); isspace(*c) && *c != '\0'; ++c) {
                };

                // face begins
                if ((strncmp(c, "outer", 5) == 0) || (strncmp(c, "OUTER", 5) == 0)) {
                    // read three vertices
                    for (i = 0; i < 3; ++i) {
                        // read line
                        c = fgets(line.data(), 100, in);
                        assert(c != nullptr);

                        // skip white-space
                        for (c = line.data(); isspace(*c) && *c != '\0'; ++c) {
                        };

                        // read x, y, z
                        sscanf(c + 6, "%f %f %f", &p[0], &p[1], &p[2]);

                        // has vector been referenced before?
                        auto it = vertex_map.find(p);
                        if (it == vertex_map.end()) {
                            // No : add vertex and remember idx/vector mapping
                            v = mesh.add_vertex((PointType) p);
                            vertices[i] = v;
                            vertex_map[p] = v;
                        } else {
                            // Yes : get index from map
                            vertices[i] = it->second;
                        }
                    }

                    // Add face only if it is not degenerated
                    if ((vertices[0] != vertices[1]) &&
                        (vertices[0] != vertices[2]) &&
                        (vertices[1] != vertices[2])) {
                        try {
                            mesh.add_face(vertices);
                        }
                        catch (const TopologyException &e) {
                            Log::Warn("Failed to add face: {}", e.what());
                        }
                    }
                }
            }
        }

        fclose(in);
        return true;
    }

    bool ReadPmp(const std::string &filepath, SurfaceMesh &mesh) {
        // open file (in binary mode)
        FILE *in = fopen(filepath.c_str(), "rb");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // how many elements?
        size_t nv{0};
        size_t ne{0};
        size_t nf{0};
        TFRead(in, nv);
        TFRead(in, ne);
        TFRead(in, nf);
        auto nh = 2 * ne;

        // texture coordinates?
        bool has_htex{false};
        TFRead(in, has_htex);

        // resize containers
        mesh.vprops_.resize(nv);
        mesh.hprops_.resize(nh);
        mesh.eprops_.resize(ne);
        mesh.fprops_.resize(nf);

        // read properties from file
        // clang-format off
        [[maybe_unused]] size_t nvc = fread((char *) mesh.vconn_.data(), sizeof(SurfaceMesh::VertexConnectivity), nv,
                                            in);
        [[maybe_unused]] size_t nhc = fread((char *) mesh.hconn_.data(), sizeof(SurfaceMesh::HalfedgeConnectivity), nh,
                                            in);
        [[maybe_unused]] size_t nfc = fread((char *) mesh.fconn_.data(), sizeof(SurfaceMesh::FaceConnectivity), nf, in);
        [[maybe_unused]] size_t np = fread((char *) mesh.vpoint_.data(), sizeof(PointType), nv, in);
        // clang-format on

        assert(nvc == nv);
        assert(nhc == nh);
        assert(nfc == nf);
        assert(np == nv);

        // read texture coordinates
        if (has_htex) {
            auto htex = mesh.halfedge_property<TexCoordType>("h:tex");
            [[maybe_unused]] size_t nhtc =
                    fread((char *) htex.data(), sizeof(TexCoordType), nh, in);
            assert(nhtc == nh);
        }

        fclose(in);
        return true;
    }

    bool Write(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), tolower);

        if (ext == ".obj") {
            return WriteObj(filepath, mesh, flags);
        } else if (ext == ".off") {
            return WriteOff(filepath, mesh, flags);
        } else if (ext == ".ply") {
            return WritePly(filepath, mesh, flags);
        } else if (ext == ".stl") {
            return WriteStl(filepath, mesh, flags);
        } else if (ext == ".pmp") {
            return WritePmp(filepath, mesh, flags);
        } else {
            return false;
        }
    }

    bool WriteObj(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags) {
        FILE *out = fopen(filepath.c_str(), "w");
        if (!out) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // check if we can write the mesh using 32-bit indices
        const auto uint_max = std::numeric_limits<uint32_t>::max();
        if (mesh.n_vertices() > uint_max) {
            Log::Error("Mesh too large to be written with 32-bit indices.");
            return false;
        }

        // comment
        fprintf(out, "# OBJ export from PMP\n");

        // write vertices
        auto points = mesh.get_vertex_property<PointType>("v:position");
        for (auto v: mesh.vertices()) {
            const PointType &p = points[v];
            fprintf(out, "v %.10f %.10f %.10f\n", p[0], p[1], p[2]);
        }

        // write normals
        auto normals = mesh.get_vertex_property<NormalType>("v:normal");
        if (normals && flags.use_vertex_normals) {
            for (auto v: mesh.vertices()) {
                const NormalType &n = normals[v];
                fprintf(out, "vn %.10f %.10f %.10f\n", n[0], n[1], n[2]);
            }
        }

        // write texture coordinates
        auto tex_coords = mesh.get_halfedge_property<TexCoordType>("h:tex");
        bool write_texcoords = tex_coords && flags.use_halfedge_texcoords;

        if (write_texcoords) {
            if (mesh.n_halfedges() > uint_max) {
                Log::Error("Mesh too large to be written with 32-bit indices.");
                return false;
            }

            for (auto h: mesh.halfedges()) {
                const TexCoordType &pt = tex_coords[h];
                fprintf(out, "vt %.10f %.10f\n", pt[0], pt[1]);
            }
        }

        // write faces
        for (auto f: mesh.faces()) {
            fprintf(out, "f");

            auto h = mesh.halfedges(f);
            for (auto v: mesh.vertices(f)) {
                auto idx = v.idx() + 1;
                if (write_texcoords) {
                    // write vertex index, texCoord index and normal index
                    fprintf(out, " %d/%d/%d", (uint32_t) idx,
                            (uint32_t) (*h).idx() + 1, (uint32_t) idx);
                    ++h;
                } else {
                    // write vertex index and normal index
                    fprintf(out, " %d//%d", (uint32_t) idx, (uint32_t) idx);
                }
            }
            fprintf(out, "\n");
        }

        fclose(out);
        return true;
    }

    template<class T>
    requires(sizeof(T) == 4)
    void WriteBinary(std::ofstream &ofs, const T &val) {
        if constexpr (std::endian::native == std::endian::little) {
            const auto u32v = std::bit_cast<uint32_t>(val);
            const auto vv = Byteswap32(u32v);
            ofs.write(reinterpret_cast<const char *>(&vv), sizeof(vv));
        } else {
            ofs.write(reinterpret_cast<const char *>(&val), sizeof(val));
        }
    }

    bool WriteOffBinary(const SurfaceMesh &mesh, const std::filesystem::path &file) {
        if constexpr (sizeof(IndexType) == 8 || sizeof(ScalarType) == 8) {
            Log::Error("Binary OFF files only support 32-bit types.");
            return false;
        }

        std::ofstream ofs(file.string());
        if (ofs.fail()) {
            Log::Error("Failed to open file: " + file.string());
            return false;
        }

        ofs << "OFF BINARY\n";
        ofs.close();
        ofs.open(file.string(), std::ios::binary | std::ios::app);

        const auto nv = static_cast<uint32_t>(mesh.n_vertices());
        const auto nf = static_cast<uint32_t>(mesh.n_faces());
        const uint32_t ne = 0;

        WriteBinary(ofs, nv);
        WriteBinary(ofs, nf);
        WriteBinary(ofs, ne);

        auto points = mesh.get_vertex_property<PointType>("v:position");
        for (auto v: mesh.vertices()) {
            const auto p = points[v];
            WriteBinary(ofs, p[0]);
            WriteBinary(ofs, p[1]);
            WriteBinary(ofs, p[2]);
        }

        for (auto f: mesh.faces()) {
            const auto valence = static_cast<uint32_t>(mesh.valence(f));
            WriteBinary(ofs, valence);
            for (auto fv: mesh.vertices(f)) {
                const uint32_t idx = fv.idx();
                WriteBinary(ofs, idx);
            }
        }
        ofs.close();
        return true;
    }


    bool WriteOff(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags) {
        if (flags.use_binary) {
            return WriteOffBinary(mesh, filepath);
        }

        // check if we can write the mesh using 32-bit indices
        if (const auto max_idx = std::numeric_limits<uint32_t>::max();
                mesh.n_vertices() > max_idx) {
            Log::Error("Mesh too large to be written with 32-bit indices.");
            return false;
        }

        FILE *out = fopen(filepath.c_str(), "w");
        if (!out) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        bool has_normals = false;
        bool has_texcoords = false;
        bool has_colors = false;

        auto normals = mesh.get_vertex_property<NormalType>("v:normal");
        auto texcoords = mesh.get_vertex_property<TexCoordType>("v:tex");
        auto colors = mesh.get_vertex_property<ColorType>("v:color");

        if (normals && flags.use_vertex_normals)
            has_normals = true;
        if (texcoords && flags.use_vertex_texcoords)
            has_texcoords = true;
        if (colors && flags.use_vertex_colors)
            has_colors = true;

        // header
        if (has_texcoords)
            fprintf(out, "ST");
        if (has_colors)
            fprintf(out, "C");
        if (has_normals)
            fprintf(out, "N");
        fprintf(out, "OFF\n%zu %zu 0\n", mesh.n_vertices(), mesh.n_faces());

        // vertices, and optionally normals and texture coordinates
        VertexProperty<PointType> points = mesh.get_vertex_property<PointType>("v:position");
        for (auto v: mesh.vertices()) {
            const PointType &p = points[v];
            fprintf(out, "%.10f %.10f %.10f", p[0], p[1], p[2]);

            if (has_normals) {
                const NormalType &n = normals[v];
                fprintf(out, " %.10f %.10f %.10f", n[0], n[1], n[2]);
            }

            if (has_colors) {
                const ColorType &c = colors[v];
                fprintf(out, " %.10f %.10f %.10f", c[0], c[1], c[2]);
            }

            if (has_texcoords) {
                const TexCoordType &t = texcoords[v];
                fprintf(out, " %.10f %.10f", t[0], t[1]);
            }

            fprintf(out, "\n");
        }

        // faces
        for (auto f: mesh.faces()) {
            auto nv = mesh.valence(f);
            fprintf(out, "%zu", nv);
            auto fv = mesh.vertices(f);
            auto fvend = fv;
            do {
                fprintf(out, " %d", (uint32_t) (*fv).idx());
            } while (++fv != fvend);
            fprintf(out, "\n");
        }

        fclose(out);
        return true;
    }

    bool WritePly(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags) {
        std::vector<std::array<double, 3>> meshVertexPositions;
        std::vector<std::array<unsigned char, 3>> meshVertexColors;
        std::vector<std::vector<size_t>> meshFaceIndices;

        meshVertexPositions.reserve(mesh.n_vertices());
        meshFaceIndices.reserve(mesh.n_faces());
        auto positions = mesh.get_vertex_property<Vector<float, 3 >>("v:position");

        for (const auto v: mesh.vertices()) {
            meshVertexPositions.push_back({positions[v][0], positions[v][1], positions[v][2]});
        }

        auto colors = mesh.get_vertex_property<Vector<float, 3>>("v:color");

        if (colors) {
            for (const auto v: mesh.vertices()) {
                meshVertexColors.push_back({(unsigned char) (colors[v][0] * 255),
                                            (unsigned char) (colors[v][1] * 255),
                                            (unsigned char) (colors[v][2] * 255)});
            }
        }
        auto triangles = mesh.get_face_property<Vector<unsigned int, 3 >>("f:indices");
        if (!triangles) {
            Log::Error("Failed to get face property: f:indices");
            return false;
        }

// Create an empty object
        happly::PLYData plyOut;

// add mesh data (elements are created automatically)
        plyOut.addVertexPositions(meshVertexPositions);
        if (colors) {
            plyOut.addVertexColors(meshVertexColors);
        }
        plyOut.addFaceIndices(meshFaceIndices);


// write the object to file
        if (flags.use_binary) {
            plyOut.write(filepath, happly::DataFormat::Binary);
        } else {
            plyOut.write(filepath, happly::DataFormat::ASCII);
        }
        return true;
    }

    inline bool WriteBinaryStl(const std::string &filepath, const SurfaceMesh &mesh) {
        std::ofstream ofs(filepath, std::ios::binary);

        // write 80 byte header
        std::string header{"Binary STL export from PMP"};
        ofs.write(header.c_str(), header.size());
        std::fill_n(std::ostream_iterator<char>(ofs), 80 - header.size(), ' ');

        //  write number of triangles
        auto n_triangles = static_cast<uint32_t>(mesh.n_faces());
        ofs.write((char *) &n_triangles, sizeof(n_triangles));

        // write normal, points, and attribute byte count
        auto normals = mesh.get_face_property<NormalType>("f:normal");
        auto points = mesh.get_vertex_property<PointType>("v:position");
        for (auto f: mesh.faces()) {
            auto n = normals[f];
            ofs.write((char *) &n[0], sizeof(float));
            ofs.write((char *) &n[1], sizeof(float));
            ofs.write((char *) &n[2], sizeof(float));

            for (auto v: mesh.vertices(f)) {
                auto p = points[v];
                ofs.write((char *) &p[0], sizeof(float));
                ofs.write((char *) &p[1], sizeof(float));
                ofs.write((char *) &p[2], sizeof(float));
            }
            ofs << "  ";
        }
        ofs.close();
        return true;
    }

    bool WriteStl(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags) {
        if (!mesh.is_triangle_mesh()) {
            Log::Error("write_stl: Not a triangle mesh.");
            return false;
        }

        auto fnormals = mesh.get_face_property<NormalType>("f:normal");
        if (!fnormals) {
            Log::Error("write_stl: No face normals present.");
            return false;
        }

        if (flags.use_binary) {
            return WriteBinaryStl(filepath, mesh);
        }

        std::ofstream ofs(filepath.c_str());
        auto points = mesh.get_vertex_property<PointType>("v:position");

        ofs << "solid stl\n";

        for (const auto &f: mesh.faces()) {
            const auto &n = fnormals[f];
            ofs << "  facet normal ";
            ofs << n[0] << " " << n[1] << " " << n[2] << "\n";
            ofs << "    outer loop\n";
            for (const auto &v: mesh.vertices(f)) {
                const auto &p = points[v];
                ofs << "      vertex ";
                ofs << p[0] << " " << p[1] << " " << p[2] << "\n";
            }
            ofs << "    endloop\n";
            ofs << "  endfacet\n";
        }
        ofs << "endsolid\n";
        ofs.close();
        return true;
    }

    bool WritePmp(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags) {
        // open file (in binary mode)
        FILE *out = fopen(filepath.c_str(), "wb");
        if (!out) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // get properties
        auto htex = mesh.get_halfedge_property<TexCoordType>("h:tex");

        // how many elements?
        auto nv = mesh.n_vertices();
        auto ne = mesh.n_edges();
        auto nh = mesh.n_halfedges();
        auto nf = mesh.n_faces();

        // write header
        TFWrite(out, nv);
        TFWrite(out, ne);
        TFWrite(out, nf);
        TFWrite(out, (bool) htex);

        // write properties to file
        // clang-format off
        fwrite((char *) mesh.vconn_.data(), sizeof(SurfaceMesh::VertexConnectivity), nv, out);
        fwrite((char *) mesh.hconn_.data(), sizeof(SurfaceMesh::HalfedgeConnectivity), nh, out);
        fwrite((char *) mesh.fconn_.data(), sizeof(SurfaceMesh::FaceConnectivity), nf, out);
        fwrite((char *) mesh.vpoint_.data(), sizeof(PointType), nv, out);
        // clang-format on

        // texture coordinates
        if (htex) {
            fwrite((char *) htex.data(), sizeof(TexCoordType), nh, out);
        }

        fclose(out);
        return true;
    }

    //------------------------------------------------------------------------------------------------------------------

    bool Read(const std::string &filepath, HalfedgeMeshInterface &mesh) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".obj") {
            return ReadObj(filepath, mesh);
        } else if (ext == ".off") {
            return ReadOff(filepath, mesh);
        } else if (ext == ".ply") {
            return ReadPly(filepath, mesh);
        } else if (ext == ".stl") {
            return ReadStl(filepath, mesh);
        } else if (ext == ".pmp") {
            return ReadPmp(filepath, mesh);
        } else {
            return false;
        }
    }

    bool ReadObj(const std::string &filepath, HalfedgeMeshInterface &mesh) {
        std::array<char, 200> s;
        float x, y, z;
        std::vector<Vertex> vertices;
        std::vector<TexCoordType> all_tex_coords; //individual texture coordinates
        std::vector<int> halfedge_tex_idx; //texture coordinates sorted for halfedges
        HalfedgeProperty<TexCoordType> tex_coords = mesh.halfedges.halfedge_property<TexCoordType>("h:tex");
        bool with_tex_coord = false;

        // open file (in ASCII mode)
        FILE *in = fopen(filepath.c_str(), "r");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // clear line once
        memset(s.data(), 0, 200);

        // parse line by line (currently only supports vertex positions & faces
        while (in && !feof(in) && fgets(s.data(), 200, in)) {
            // comment
            if (s[0] == '#' || isspace(s[0]))
                continue;

                // vertex
            else if (strncmp(s.data(), "v ", 2) == 0) {
                if (sscanf(s.data(), "v %f %f %f", &x, &y, &z)) {
                    mesh.add_vertex(PointType(x, y, z));
                }
            }

                // normal
            else if (strncmp(s.data(), "vn ", 3) == 0) {
                if (sscanf(s.data(), "vn %f %f %f", &x, &y, &z)) {
                    // problematic as it can be either a vertex property when interpolated
                    // or a halfedge property for hard edges
                }
            }

                // texture coordinate
            else if (strncmp(s.data(), "vt ", 3) == 0) {
                if (sscanf(s.data(), "vt %f %f", &x, &y)) {
                    all_tex_coords.emplace_back(x, y);
                }
            }

                // face
            else if (strncmp(s.data(), "f ", 2) == 0) {
                int component(0);
                bool end_of_vertex(false);
                char *p0, *p1(s.data() + 1);

                vertices.clear();
                halfedge_tex_idx.clear();

                // skip white-spaces
                while (*p1 == ' ')
                    ++p1;

                while (p1) {
                    p0 = p1;

                    // overwrite next separator

                    // skip '/', '\n', ' ', '\0', '\r' <-- don't forget Windows
                    while (*p1 != '/' && *p1 != '\r' && *p1 != '\n' && *p1 != ' ' &&
                           *p1 != '\0')
                        ++p1;

                    // detect end of vertex
                    if (*p1 != '/') {
                        end_of_vertex = true;
                    }

                    // replace separator by '\0'
                    if (*p1 != '\0') {
                        *p1 = '\0';
                        p1++; // point to next token
                    }

                    // detect end of line and break
                    if (*p1 == '\0' || *p1 == '\n') {
                        p1 = nullptr;
                    }

                    // read next vertex component
                    if (*p0 != '\0') {
                        switch (component) {
                            case 0: // vertex
                            {
                                int idx = atoi(p0);
                                if (idx < 0)
                                    idx = mesh.vertices.size() + idx + 1;
                                vertices.emplace_back(idx - 1);
                                break;
                            }
                            case 1: // texture coord
                            {
                                int idx = atoi(p0) - 1;
                                halfedge_tex_idx.push_back(idx);
                                with_tex_coord = true;
                                break;
                            }
                            case 2: // normal
                                break;
                        }
                    }

                    ++component;

                    if (end_of_vertex) {
                        component = 0;
                        end_of_vertex = false;
                    }
                }

                Face f;
                try {
                    f = mesh.add_face(vertices);
                } catch (const TopologyException &e) {
                    Log::Warn("Failed to add face: {}", e.what());
                }

                // add texture coordinates
                if (with_tex_coord && f.is_valid()) {
                    auto h_fit = mesh.get_halfedges(f);
                    auto h_end = h_fit;
                    unsigned v_idx = 0;
                    do {
                        tex_coords[*h_fit] =
                                all_tex_coords.at(halfedge_tex_idx.at(v_idx));
                        ++v_idx;
                        ++h_fit;
                    } while (h_fit != h_end);
                }
            }
            // clear line
            memset(s.data(), 0, 200);
        }

        // if there are no textures, delete texture property!
        if (!with_tex_coord) {
            mesh.halfedges.remove_halfedge_property(tex_coords);
        }

        fclose(in);
        return true;
    }


    void ReadOffAscii(HalfedgeMeshInterface &mesh, FILE *in, const bool has_normals,
                      const bool has_texcoords, const bool has_colors,
                      char *first_line) {
        std::array<char, 1000> line;
        char *lp = first_line;
        int nc;
        long int i, j, idx;
        long int nv, nf, ne;
        float x, y, z, r, g, b;
        Vertex v;

        // properties
        VertexProperty<NormalType> normals;
        VertexProperty<TexCoordType> texcoords;
        VertexProperty<ColorType> colors;
        if (has_normals)
            normals = mesh.vertices.vertex_property<NormalType>("v:normal");
        if (has_texcoords)
            texcoords = mesh.vertices.vertex_property<TexCoordType>("v:tex");
        if (has_colors)
            colors = mesh.vertices.vertex_property<ColorType>("v:color");

        // read line, but skip comment lines
        while (lp && (lp[0] == '#' || lp[0] == '\n' || lp[0] == '\r')) {
            lp = fgets(line.data(), 1000, in);
        }

        // #Vertices, #Faces, #Edges
        auto items = sscanf(lp, "%ld %ld %ld\n", &nv, &nf, &ne);

        if (items < 3 || ne < 0) {
            Log::Error("Failed to parse OFF header");
            return;
        }

        if (nv < 1) {
            Log::Error("Off file has no vertices");
            return;
        }


        mesh.reserve(nv, std::max(3 * nv, ne), nf);

        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nv && !feof(in); ++i) {
            // read line, but skip comment lines
            do {
                lp = fgets(line.data(), 1000, in);
            } while (lp && (lp[0] == '#' || lp[0] == '\n'));
            lp = line.data();

            // position
            items = sscanf(lp, "%f %f %f%n", &x, &y, &z, &nc);
            assert(items == 3);
            v = mesh.add_vertex(PointType(x, y, z));
            lp += nc;

            // normal
            if (has_normals) {
                if (sscanf(lp, "%f %f %f%n", &x, &y, &z, &nc) == 3) {
                    normals[v] = NormalType(x, y, z);
                }
                lp += nc;
            }

            // color
            if (has_colors) {
                if (sscanf(lp, "%f %f %f%n", &r, &g, &b, &nc) == 3) {
                    if (r > 1.0f || g > 1.0f || b > 1.0f) {
                        r /= 255.0f;
                        g /= 255.0f;
                        b /= 255.0f;
                    }
                    colors[v] = ColorType(r, g, b);
                }
                lp += nc;
            }

            // tex coord
            if (has_texcoords) {
                items = sscanf(lp, "%f %f%n", &x, &y, &nc);
                assert(items == 2);
                texcoords[v][0] = x;
                texcoords[v][1] = y;
                lp += nc;
            }
        }

        if (nf < 1) {
            Log::Warn("Off file has no faces");
            return;
        }

        // read faces: #N v[1] v[2] ... v[n-1]
        std::vector<Vertex> vertices;
        for (i = 0; i < nf; ++i) {
            // read line, but skip comment lines
            do {
                lp = fgets(line.data(), 1000, in);
            } while (lp && (lp[0] == '#' || lp[0] == '\n'));
            lp = line.data();

            // #vertices
            items = sscanf(lp, "%ld%n", &nv, &nc);
            assert(items == 1);
            if (nv < 1)
                throw IOException("Invalid index count");
            vertices.resize(nv);
            lp += nc;

            // indices
            for (j = 0; j < nv; ++j) {
                items = sscanf(lp, "%ld%n", &idx, &nc);
                assert(items == 1);
                if (idx < 0)
                    throw IOException("Invalid index");
                vertices[j] = Vertex(idx);
                lp += nc;
            }
            try {
                mesh.add_face(vertices);
            }
            catch (const TopologyException &e) {
                Log::Warn("Failed to add face: {}", e.what());
            }
        }
    }

    void ReadOffBinary(HalfedgeMeshInterface &mesh, FILE *in, const bool has_normals,
                       const bool has_texcoords, const bool has_colors,
                       const std::filesystem::path &file) {
        uint32_t i, j, idx(0);
        uint32_t nv(0), nf(0), ne(0);
        PointType p, n;
        TexCoordType t;
        Vertex v;

        // binary cannot (yet) read colors
        if (has_colors)
            throw IOException("Colors not supported for binary OFF file.");

        // properties
        VertexProperty<NormalType> normals;
        VertexProperty<TexCoordType> texcoords;
        if (has_normals)
            normals = mesh.vertices.vertex_property<NormalType>("v:normal");
        if (has_texcoords)
            texcoords = mesh.vertices.vertex_property<TexCoordType>("v:tex");

        // #Vertices, #Faces, #Edges
        ReadBinary(in, nv);

        // Check for little endian encoding used by previous versions.
        // Swap the ordering if the total file size is smaller than the size
        // required to store all vertex coordinates.
        auto file_size = std::filesystem::file_size(file);
        bool swap = file_size < nv * 3 * 4 ? true : false;
        if (swap)
            nv = Byteswap32(nv);

        ReadBinary(in, nf, swap);
        ReadBinary(in, ne, swap);


        mesh.reserve(nv, std::max(3 * nv, ne), nf);

        // read vertices: pos [normal] [color] [texcoord]
        for (i = 0; i < nv && !feof(in); ++i) {
            // position
            ReadBinary(in, p[0], swap);
            ReadBinary(in, p[1], swap);
            ReadBinary(in, p[2], swap);
            v = mesh.add_vertex((PointType) p);

            // normal
            if (has_normals) {
                ReadBinary(in, n[0], swap);
                ReadBinary(in, n[1], swap);
                ReadBinary(in, n[2], swap);
                normals[v] = (NormalType) n;
            }

            // tex coord
            if (has_texcoords) {
                ReadBinary(in, t[0], swap);
                ReadBinary(in, t[1], swap);
                texcoords[v][0] = t[0];
                texcoords[v][1] = t[1];
            }
        }

        // read faces: #N v[1] v[2] ... v[n-1]
        std::vector<Vertex> vertices;
        for (i = 0; i < nf; ++i) {
            ReadBinary(in, nv, swap);
            vertices.resize(nv);
            for (j = 0; j < nv; ++j) {
                ReadBinary(in, idx, swap);
                vertices[j] = Vertex(idx);
            }
            try {
                mesh.add_face(vertices);
            }
            catch (const TopologyException &e) {
                Log::Warn("Failed to add face: {}", e.what());
            }
        }
    }

    bool ReadOff(const std::string &filepath, HalfedgeMeshInterface &mesh) {
        std::array<char, 200> line;
        bool has_texcoords = false;
        bool has_normals = false;
        bool has_colors = false;
        bool has_hcoords = false;
        bool has_dim = false;
        bool is_binary = false;

        // open file (in ASCII mode)
        FILE *in = fopen(filepath.c_str(), "r");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // read header: [ST][C][N][4][n]OFF BINARY
        char *c = fgets(line.data(), 200, in);
        assert(c != nullptr);
        c = line.data();
        if (c[0] == 'S' && c[1] == 'T') {
            has_texcoords = true;
            c += 2;
        }
        if (c[0] == 'C') {
            has_colors = true;
            ++c;
        }
        if (c[0] == 'N') {
            has_normals = true;
            ++c;
        }
        if (c[0] == '4') {
            has_hcoords = true;
            ++c;
        }
        if (c[0] == 'n') {
            has_dim = true;
            ++c;
        }
        if (strncmp(c, "OFF", 3) != 0) {
            fclose(in);
            Log::Error("Failed to parse OFF header");
            return false;
        }
        c += 3;
        if (c[0] == ' ')
            ++c;
        if (strncmp(c, "BINARY", 6) == 0) {
            is_binary = true;
            c += 6;
        }
        if (c[0] == ' ')
            ++c;

        if (has_hcoords) {
            fclose(in);
            Log::Error("Error: Homogeneous coordinates not supported.");
            return false;
        }
        if (has_dim) {
            fclose(in);
            Log::Error("Error: vertex dimension != 3 not supported");
            return false;
        }

        // if binary: reopen file in binary mode
        if (is_binary) {
            fclose(in);
            in = fopen(filepath.c_str(), "rb");
            c = fgets(line.data(), 200, in);
            assert(c != nullptr);
        }

        // read as ASCII or binary
        if (is_binary) {
            ReadOffBinary(mesh, in, has_normals, has_texcoords, has_colors, filepath);
        } else {
            ReadOffAscii(mesh, in, has_normals, has_texcoords, has_colors, c);
        }

        fclose(in);
        return true;;
    }

    bool ReadPly(const std::string &filepath, HalfedgeMeshInterface &mesh) {
        happly::PLYData plyIn(filepath);

        if (!plyIn.hasElement("vertex") ||
            !plyIn.getElement("vertex").hasProperty("x") ||
            !plyIn.getElement("vertex").hasProperty("y") ||
            !plyIn.getElement("vertex").hasProperty("z")) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
        std::vector<std::array<unsigned char, 3>> vCol;
        if (plyIn.getElement("vertex").hasProperty("red")) {
            vCol = plyIn.getVertexColors();
        }
        std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();

        auto colors = mesh.vertices.vertex_property<Vector<float, 3>>("v:color");


        mesh.vertices.reserve(vPos.size());
        for (const auto &point: vPos) {
            mesh.add_vertex(Vector<double, 3>(point.data()).cast<float>());
        }

        if (!vCol.empty()) {
            for (const auto &v: mesh.vertices) {
                const Eigen::Vector<double, 3> color(vCol[v.idx()][0] / 255.0, vCol[v.idx()][1] / 255.0,
                                                     vCol[v.idx()][2] / 255.0);
                colors[v] = color.cast<float>();
            }
        } else {
            mesh.vertices.remove(colors);
        }

        mesh.faces.reserve(fInd.size());
        for (const auto &face: fInd) {
            mesh.add_face({Vertex(face[0]), Vertex(face[1]), Vertex(face[2])});
        }

        return mesh.faces.size() > 0;
    }

    bool ReadStl(const std::string &filepath, HalfedgeMeshInterface &mesh) {
        std::array<char, 100> line;
        uint32_t i, nT(0);
        vec3 p;
        Vertex v;
        std::vector<Vertex> vertices(3);

        CompareVec3 comp;
        std::map<vec3, Vertex, CompareVec3> vertex_map(comp);

        // open file (in ASCII mode)
        FILE *in = fopen(filepath.c_str(), "r");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // determine if the file is a binary STL file
        auto is_binary = [&]() {
            [[maybe_unused]] auto c = fgets(line.data(), 6, in);

            // if the file does *not* start with "solid" we have a binary file
            if ((strncmp(line.data(), "SOLID", 5) != 0) &&
                (strncmp(line.data(), "solid", 5) != 0)) {
                return true;
            }

            // otherwise check if file size matches number of triangles
            auto fp = fopen(filepath.c_str(), "rb");
            if (!fp) {
                Log::Error("Failed to open file: " + filepath);
                return false;
            }

            // skip header
            [[maybe_unused]] auto n_items = fread(line.data(), 1, 80, fp);

            // read number of triangles
            uint32_t n_triangles{0};
            TFRead(fp, n_triangles);

            // get file size minus header and element count
            fseek(fp, 0L, SEEK_END);
            auto size = ftell(fp);
            size -= 84;
            fclose(fp);

            // for each triangle we should have 4*12+2 bytes:
            // normal, x,y,z, attribute byte count
            auto predicted = (4 * 12 + 2) * n_triangles;

            return size == predicted;
        };



        // parse binary STL
        if (is_binary()) {
            // re-open file in binary mode
            fclose(in);
            in = fopen(filepath.c_str(), "rb");
            if (!in) {
                Log::Error("Failed to open file: " + filepath);
                return false;
            }

            // skip dummy header
            [[maybe_unused]] auto n_items = fread(line.data(), 1, 80, in);
            assert(n_items > 0);

            // read number of triangles
            TFRead(in, nT);

            // read triangles
            while (nT) {
                // skip triangle normal
                n_items = fread(line.data(), 1, 12, in);
                assert(n_items > 0);

                // triangle's vertices
                for (i = 0; i < 3; ++i) {
                    TFRead(in, p);

                    // has vector been referenced before?
                    auto it = vertex_map.find(p);
                    if (it == vertex_map.end()) {
                        // No : add vertex and remember idx/vector mapping
                        v = mesh.add_vertex((PointType) p);
                        vertices[i] = v;
                        vertex_map[p] = v;
                    } else {
                        // Yes : get index from map
                        vertices[i] = it->second;
                    }
                }

                // Add face only if it is not degenerated
                if ((vertices[0] != vertices[1]) && (vertices[0] != vertices[2]) &&
                    (vertices[1] != vertices[2])) {
                    try {
                        mesh.add_face(vertices);
                    }
                    catch (const TopologyException &e) {
                        Log::Warn("Failed to add face: {}", e.what());
                    }
                }

                n_items = fread(line.data(), 1, 2, in);
                assert(n_items > 0);

                --nT;
            }
        }

            // parse ASCII STL
        else {
            char *c{nullptr};

            // parse line by line
            while (in && !feof(in) && fgets(line.data(), 100, in)) {
                // skip white-space
                for (c = line.data(); isspace(*c) && *c != '\0'; ++c) {
                };

                // face begins
                if ((strncmp(c, "outer", 5) == 0) || (strncmp(c, "OUTER", 5) == 0)) {
                    // read three vertices
                    for (i = 0; i < 3; ++i) {
                        // read line
                        c = fgets(line.data(), 100, in);
                        assert(c != nullptr);

                        // skip white-space
                        for (c = line.data(); isspace(*c) && *c != '\0'; ++c) {
                        };

                        // read x, y, z
                        sscanf(c + 6, "%f %f %f", &p[0], &p[1], &p[2]);

                        // has vector been referenced before?
                        auto it = vertex_map.find(p);
                        if (it == vertex_map.end()) {
                            // No : add vertex and remember idx/vector mapping
                            v = mesh.add_vertex((PointType) p);
                            vertices[i] = v;
                            vertex_map[p] = v;
                        } else {
                            // Yes : get index from map
                            vertices[i] = it->second;
                        }
                    }

                    // Add face only if it is not degenerated
                    if ((vertices[0] != vertices[1]) &&
                        (vertices[0] != vertices[2]) &&
                        (vertices[1] != vertices[2])) {
                        try {
                            mesh.add_face(vertices);
                        }
                        catch (const TopologyException &e) {
                            Log::Warn("Failed to add face: {}", e.what());
                        }
                    }
                }
            }
        }

        fclose(in);
        return true;
    }

    bool ReadPmp(const std::string &filepath, HalfedgeMeshInterface &mesh) {
        // open file (in binary mode)
        FILE *in = fopen(filepath.c_str(), "rb");
        if (!in) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // how many elements?
        size_t nv{0};
        size_t ne{0};
        size_t nf{0};
        TFRead(in, nv);
        TFRead(in, ne);
        TFRead(in, nf);
        auto nh = 2 * ne;

        // texture coordinates?
        bool has_htex{false};
        TFRead(in, has_htex);

        // resize containers
        mesh.vertices.resize(nv);
        mesh.halfedges.resize(nh);
        mesh.edges.resize(ne);
        mesh.faces.resize(nf);



        // read properties from file
        // clang-format off
        [[maybe_unused]] size_t nvc = fread((char *) mesh.vconnectivity.data(), sizeof(Halfedge), nv,
                                            in);
        [[maybe_unused]] size_t nhc = fread((char *) mesh.hconnectivity.data(),
                                            sizeof(HalfedgeMeshInterface::HalfedgeConnectivity), nh,
                                            in);
        [[maybe_unused]] size_t nfc = fread((char *) mesh.fconnectivity.data(), sizeof(Halfedge), nf, in);
        [[maybe_unused]] size_t np = fread((char *) mesh.vpoint.data(), sizeof(PointType), nv, in);
        // clang-format on

        assert(nvc == nv);
        assert(nhc == nh);
        assert(nfc == nf);
        assert(np == nv);

        // read texture coordinates
        if (has_htex) {
            auto htex = mesh.halfedges.halfedge_property<TexCoordType>("h:tex");
            [[maybe_unused]] size_t nhtc =
                    fread((char *) htex.data(), sizeof(TexCoordType), nh, in);
            assert(nhtc == nh);
        }

        fclose(in);
        return true;
    }

    bool Write(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), tolower);

        if (ext == ".obj") {
            return WriteObj(filepath, mesh, flags);
        } else if (ext == ".off") {
            return WriteOff(filepath, mesh, flags);
        } else if (ext == ".ply") {
            return WritePly(filepath, mesh, flags);
        } else if (ext == ".stl") {
            return WriteStl(filepath, mesh, flags);
        } else if (ext == ".pmp") {
            return WritePmp(filepath, mesh, flags);
        } else {
            return false;
        }
    }

    bool WriteObj(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags) {
        FILE *out = fopen(filepath.c_str(), "w");
        if (!out) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // check if we can write the mesh using 32-bit indices
        const auto uint_max = std::numeric_limits<uint32_t>::max();
        if (mesh.vertices.n_vertices() > uint_max) {
            Log::Error("Mesh too large to be written with 32-bit indices.");
            return false;
        }

        // comment
        fprintf(out, "# OBJ export from PMP\n");

        // write vertices
        auto points = mesh.vertices.get_vertex_property<PointType>("v:position");
        for (auto v: mesh.vertices) {
            const PointType &p = points[v];
            fprintf(out, "v %.10f %.10f %.10f\n", p[0], p[1], p[2]);
        }

        // write normals
        auto normals = mesh.vertices.get_vertex_property<NormalType>("v:normal");
        if (normals && flags.use_vertex_normals) {
            for (auto v: mesh.vertices) {
                const NormalType &n = normals[v];
                fprintf(out, "vn %.10f %.10f %.10f\n", n[0], n[1], n[2]);
            }
        }

        // write texture coordinates
        auto tex_coords = mesh.halfedges.get_halfedge_property<TexCoordType>("h:tex");
        bool write_texcoords = tex_coords && flags.use_halfedge_texcoords;

        if (write_texcoords) {
            if (mesh.halfedges.n_halfedges() > uint_max) {
                Log::Error("Mesh too large to be written with 32-bit indices.");
                return false;
            }

            for (auto h: mesh.halfedges) {
                const TexCoordType &pt = tex_coords[h];
                fprintf(out, "vt %.10f %.10f\n", pt[0], pt[1]);
            }
        }



        // write faces
        for (auto f: mesh.faces) {
            fprintf(out, "f");

            auto h = mesh.get_halfedges(f);
            for (auto v: mesh.get_vertices(f)) {
                auto idx = v.idx() + 1;
                if (write_texcoords) {
                    // write vertex index, texCoord index and normal index
                    fprintf(out, " %d/%d/%d", (uint32_t) idx,
                            (uint32_t) (*h).idx() + 1, (uint32_t) idx);
                    ++h;
                } else {
                    // write vertex index and normal index
                    fprintf(out, " %d//%d", (uint32_t) idx, (uint32_t) idx);
                }
            }
            fprintf(out, "\n");
        }

        fclose(out);
        return true;
    }

    bool WriteOffBinary(const HalfedgeMeshInterface &mesh, const std::filesystem::path &file) {
        if constexpr (sizeof(IndexType) == 8 || sizeof(ScalarType) == 8) {
            Log::Error("Binary OFF files only support 32-bit types.");
            return false;
        }

        std::ofstream ofs(file.string());
        if (ofs.fail()) {
            Log::Error("Failed to open file: " + file.string());
            return false;
        }

        ofs << "OFF BINARY\n";
        ofs.close();
        ofs.open(file.string(), std::ios::binary | std::ios::app);

        const auto nv = static_cast<uint32_t>(mesh.vertices.n_vertices());
        const auto nf = static_cast<uint32_t>(mesh.faces.n_faces());
        const uint32_t ne = 0;

        WriteBinary(ofs, nv);
        WriteBinary(ofs, nf);
        WriteBinary(ofs, ne);

        auto points = mesh.vertices.get_vertex_property<PointType>("v:position");
        for (auto v: mesh.vertices) {
            const auto p = points[v];
            WriteBinary(ofs, p[0]);
            WriteBinary(ofs, p[1]);
            WriteBinary(ofs, p[2]);
        }

        for (auto f: mesh.faces) {
            const auto valence = static_cast<uint32_t>(mesh.valence(f));
            WriteBinary(ofs, valence);
            for (auto fv: mesh.get_vertices(f)) {
                const uint32_t idx = fv.idx();
                WriteBinary(ofs, idx);
            }
        }
        ofs.close();
        return true;
    }


    bool WriteOff(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags) {
        if (flags.use_binary) {
            return WriteOffBinary(mesh, filepath);
        }

        // check if we can write the mesh using 32-bit indices
        if (const auto max_idx = std::numeric_limits<uint32_t>::max();
                mesh.vertices.n_vertices() > max_idx) {
            Log::Error("Mesh too large to be written with 32-bit indices.");
            return false;
        }

        FILE *out = fopen(filepath.c_str(), "w");
        if (!out) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        bool has_normals = false;
        bool has_texcoords = false;
        bool has_colors = false;

        auto normals = mesh.get_vertex_property<NormalType>("v:normal");
        auto texcoords = mesh.get_vertex_property<TexCoordType>("v:tex");
        auto colors = mesh.get_vertex_property<ColorType>("v:color");

        if (normals && flags.use_vertex_normals)
            has_normals = true;
        if (texcoords && flags.use_vertex_texcoords)
            has_texcoords = true;
        if (colors && flags.use_vertex_colors)
            has_colors = true;

        // header
        if (has_texcoords)
            fprintf(out, "ST");
        if (has_colors)
            fprintf(out, "C");
        if (has_normals)
            fprintf(out, "N");
        fprintf(out, "OFF\n%zu %zu 0\n", mesh.vertices.n_vertices(), mesh.faces.n_faces());

        // vertices, and optionally normals and texture coordinates
        VertexProperty<PointType> points = mesh.get_vertex_property<PointType>("v:position");
        for (auto v: mesh.vertices) {
            const PointType &p = points[v];
            fprintf(out, "%.10f %.10f %.10f", p[0], p[1], p[2]);

            if (has_normals) {
                const NormalType &n = normals[v];
                fprintf(out, " %.10f %.10f %.10f", n[0], n[1], n[2]);
            }

            if (has_colors) {
                const ColorType &c = colors[v];
                fprintf(out, " %.10f %.10f %.10f", c[0], c[1], c[2]);
            }

            if (has_texcoords) {
                const TexCoordType &t = texcoords[v];
                fprintf(out, " %.10f %.10f", t[0], t[1]);
            }

            fprintf(out, "\n");
        }

        // faces
        for (auto f: mesh.faces) {
            auto nv = mesh.valence(f);
            fprintf(out, "%zu", nv);
            auto fv = mesh.get_vertices(f);
            auto fvend = fv;
            do {
                fprintf(out, " %d", (uint32_t) (*fv).idx());
            } while (++fv != fvend);
            fprintf(out, "\n");
        }

        fclose(out);
        return true;
    }

    bool WritePly(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags) {
        std::vector<std::array<double, 3>> meshVertexPositions;
        std::vector<std::array<unsigned char, 3>> meshVertexColors;
        std::vector<std::vector<size_t>> meshFaceIndices;

        meshVertexPositions.reserve(mesh.vertices.n_vertices());
        meshFaceIndices.reserve(mesh.faces.n_faces());
        auto positions = mesh.get_vertex_property<Vector<float, 3 >>("v:position");

        for (const auto v: mesh.vertices) {
            meshVertexPositions.push_back({positions[v][0], positions[v][1], positions[v][2]});
        }

        auto colors = mesh.get_vertex_property<Vector<float, 3>>("v:color");

        if (colors) {
            for (const auto v: mesh.vertices) {
                meshVertexColors.push_back({(unsigned char) (colors[v][0] * 255),
                                            (unsigned char) (colors[v][1] * 255),
                                            (unsigned char) (colors[v][2] * 255)});
            }
        }
        auto triangles = mesh.get_face_property<Vector<unsigned int, 3 >>("f:indices");
        if (!triangles) {
            Log::Error("Failed to get face property: f:indices");
            return false;
        }

// Create an empty object
        happly::PLYData plyOut;

// add mesh data (elements are created automatically)
        plyOut.addVertexPositions(meshVertexPositions);
        if (colors) {
            plyOut.addVertexColors(meshVertexColors);
        }
        plyOut.addFaceIndices(meshFaceIndices);


// write the object to file
        if (flags.use_binary) {
            plyOut.write(filepath, happly::DataFormat::Binary);
        } else {
            plyOut.write(filepath, happly::DataFormat::ASCII);
        }
        return true;
    }

    inline bool WriteBinaryStl(const std::string &filepath, const HalfedgeMeshInterface &mesh) {
        std::ofstream ofs(filepath, std::ios::binary);

        // write 80 byte header
        std::string header{"Binary STL export from PMP"};
        ofs.write(header.c_str(), header.size());
        std::fill_n(std::ostream_iterator<char>(ofs), 80 - header.size(), ' ');

        //  write number of triangles
        auto n_triangles = static_cast<uint32_t>(mesh.faces.n_faces());
        ofs.write((char *) &n_triangles, sizeof(n_triangles));

        // write normal, points, and attribute byte count
        auto normals = mesh.get_face_property<NormalType>("f:normal");
        auto points = mesh.get_vertex_property<PointType>("v:position");
        for (auto f: mesh.faces) {
            auto n = normals[f];
            ofs.write((char *) &n[0], sizeof(float));
            ofs.write((char *) &n[1], sizeof(float));
            ofs.write((char *) &n[2], sizeof(float));

            for (auto v: mesh.get_vertices(f)) {
                auto p = points[v];
                ofs.write((char *) &p[0], sizeof(float));
                ofs.write((char *) &p[1], sizeof(float));
                ofs.write((char *) &p[2], sizeof(float));
            }
            ofs << "  ";
        }
        ofs.close();
        return true;
    }

    bool WriteStl(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags) {
        if (!mesh.is_triangle_mesh()) {
            Log::Error("write_stl: Not a triangle mesh.");
            return false;
        }

        auto fnormals = mesh.get_face_property<NormalType>("f:normal");
        if (!fnormals) {
            Log::Error("write_stl: No face normals present.");
            return false;
        }

        if (flags.use_binary) {
            return WriteBinaryStl(filepath, mesh);
        }

        std::ofstream ofs(filepath.c_str());
        auto points = mesh.get_vertex_property<PointType>("v:position");

        ofs << "solid stl\n";

        for (const auto &f: mesh.faces) {
            const auto &n = fnormals[f];
            ofs << "  facet normal ";
            ofs << n[0] << " " << n[1] << " " << n[2] << "\n";
            ofs << "    outer loop\n";
            for (const auto &v: mesh.get_vertices(f)) {
                const auto &p = points[v];
                ofs << "      vertex ";
                ofs << p[0] << " " << p[1] << " " << p[2] << "\n";
            }
            ofs << "    endloop\n";
            ofs << "  endfacet\n";
        }
        ofs << "endsolid\n";
        ofs.close();
        return true;
    }

    bool WritePmp(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags) {
        // open file (in binary mode)
        FILE *out = fopen(filepath.c_str(), "wb");
        if (!out) {
            Log::Error("Failed to open file: " + filepath);
            return false;
        }

        // get properties
        auto htex = mesh.get_halfedge_property<TexCoordType>("h:tex");

        // how many elements?
        auto nv = mesh.vertices.n_vertices();
        auto ne = mesh.edges.n_edges();
        auto nh = mesh.halfedges.n_halfedges();
        auto nf = mesh.faces.n_faces();

        // write header
        TFWrite(out, nv);
        TFWrite(out, ne);
        TFWrite(out, nf);
        TFWrite(out, (bool) htex);

        // write properties to file
        // clang-format off
        fwrite((char *) mesh.vconnectivity.data(), sizeof(Halfedge), nv, out);
        fwrite((char *) mesh.hconnectivity.data(), sizeof(HalfedgeMeshInterface::HalfedgeConnectivity), nh, out);
        fwrite((char *) mesh.fconnectivity.data(), sizeof(Halfedge), nf, out);
        fwrite((char *) mesh.vpoint.data(), sizeof(PointType), nv, out);
        // clang-format on

        // texture coordinates
        if (htex) {
            fwrite((char *) htex.data(), sizeof(TexCoordType), nh, out);
        }

        fclose(out);
        return true;
    }
}