//
// Created by alex on 13.08.24.
//

#include "PointCloudIo.h"
#include "Utils.h"
#include "PropertyEigenMap.h"
#include "happly.h"

#include <filesystem>
#include <fstream>

namespace Bcg {
    bool Read(const std::string &filepath, PointCloud &pc) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".csv") {
            return ReadCsv(filepath, pc);
        } else if (ext == ".pts") {
            return ReadPts(filepath, pc);
        } else if (ext == ".xyz") {
            return ReadXyz(filepath, pc);
        } else if (ext == ".ply") {
            return ReadPly(filepath, pc);
        } else {
            return false;
        }
    }

    bool ReadCsv(const std::string &filepath, PointCloud &pc) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<ScalarType> numbers = ParseNumbers(txt, rows);
        cols = numbers.size() / rows;
        assert(cols == 7);
        auto mapped = Map(numbers, rows, cols);
        auto colors = pc.interface.vertex_property<ColorType>("v:color", ColorType(0.0f));
        auto intensities = pc.interface.vertex_property<ScalarType>("v:intensity", 1);

        pc.data.vertices.resize(rows);
        Map(pc.interface.vpoint.vector()) = mapped.block(0, 0, rows, 3);
        Map(intensities.vector()) = mapped.block(0, 3, rows, 1);
        Map(colors.vector()) = mapped.block(0, 4, rows, 3);
        return true;
    }

    bool ReadPts(const std::string &filepath, PointCloud &pc) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<ScalarType> numbers = ParseNumbers(txt, rows);
        cols = numbers.size() / rows;
        assert(cols == 7);
        auto mapped = Map(numbers, rows, cols);
        auto colors = pc.interface.vertex_property<ColorType>("v:color", ColorType(0.0f));
        auto intensities = pc.interface.vertex_property<ScalarType>("v:intensity", 1);

        pc.data.vertices.resize(rows);
        Map(pc.interface.vpoint.vector()) = mapped.block(0, 0, rows, 3).transpose();
        Map(intensities.vector()) = mapped.block(0, 3, rows, 1);
        Map(colors.vector()) = mapped.block(0, 4, rows, 3).transpose();
        return true;
    }

    bool ReadXyz(const std::string &filepath, PointCloud &pc) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<float> numbers = ParseNumbers(txt, cols, "LH");
        rows = numbers.size() / cols;

        auto mapped = Map(numbers, rows, cols);

        pc.data.vertices.resize(cols);
        Map(pc.interface.vpoint.vector()) = mapped.block(0, 0, 3, cols);

        if (cols == 6) {
            auto colors = pc.interface.vertex_property<ColorType>("v:color", ColorType(0.0f));
            Map(colors.vector()) = mapped.block(3, 0, 3, cols);
        }
        return true;
    }


    bool ReadPly(const std::string &filepath, PointCloud &pc) {
        return ReadPly(filepath, pc.interface);
    }

    bool Write(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), tolower);

        if (ext == ".csv") {
            return WriteCsv(filepath, pc, flags);
        } else if (ext == ".pts") {
            return WritePts(filepath, pc, flags);
        } else if (ext == ".xyz") {
            return WriteXyz(filepath, pc, flags);
        } else {
            return false;
        }
    }

    bool WriteCsv(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pc.interface.get_vertex_property<PointType>("v:point");
        auto intensities = pc.interface.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pc.interface.get_vertex_property<ColorType>("v:color");
            for (const auto v: pc.data.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << " " << Map(colors[v]).transpose() <<
                        std::endl;
            }
        } else {
            for (const auto v: pc.data.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << std::endl;
            }
        }
        return true;
    }

    bool WritePts(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pc.interface.get_vertex_property<PointType>("v:point");
        auto intensities = pc.interface.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pc.interface.get_vertex_property<ColorType>("v:color");
            for (const auto v: pc.data.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << " " << Map(colors[v]).transpose() <<
                        std::endl;
            }
        } else {
            for (const auto v: pc.data.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << std::endl;
            }
        }
        return true;
    }

    bool WriteXyz(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pc.interface.get_vertex_property<PointType>("v:point");
        auto intensities = pc.interface.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pc.interface.get_vertex_property<ColorType>("v:color");
            for (const auto v: pc.data.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << " " << Map(colors[v]).transpose() <<
                        std::endl;
            }
        } else {
            for (const auto v: pc.data.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << std::endl;
            }
        }
        return true;
    }

    //------------------------------------------------------------------------------------------------------------------


    bool Read(const std::string &filepath, PointCloudInterface &pci) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".csv") {
            return ReadCsv(filepath, pci);
        } else if (ext == ".pts") {
            return ReadPts(filepath, pci);
        } else if (ext == ".xyz") {
            return ReadXyz(filepath, pci);
        } else {
            return false;
        }
    }

    bool ReadCsv(const std::string &filepath, PointCloudInterface &pci) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<ScalarType> numbers = ParseNumbers(txt, rows);
        cols = numbers.size() / rows;
        assert(cols == 7);
        auto mapped = Map(numbers, rows, cols);

        auto positions = pci.vertices.vertex_property<PointType>("v:point", PointType(0.0f));
        auto colors = pci.vertices.vertex_property<ColorType>("v:color", ColorType(0.0f));
        auto intensities = pci.vertices.vertex_property<ScalarType>("v:intensity", 1);

        pci.vertices.resize(rows);
        Map(positions.vector()) = mapped.block(0, 0, rows, 3);
        Map(intensities.vector()) = mapped.block(0, 3, rows, 1);
        Map(colors.vector()) = mapped.block(0, 4, rows, 3);
        return true;
    }

    bool ReadPts(const std::string &filepath, PointCloudInterface &pci) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<ScalarType> numbers = ParseNumbers(txt, rows);
        cols = numbers.size() / rows;
        assert(cols == 7);
        auto mapped = Map(numbers, rows, cols);

        auto positions = pci.vertices.vertex_property<PointType>("v:point", PointType(0.0f));
        auto colors = pci.vertices.vertex_property<ColorType>("v:color", ColorType(0.0f));
        auto intensities = pci.vertices.vertex_property<ScalarType>("v:intensity", 1);

        pci.vertices.resize(rows);
        Map(positions.vector()) = mapped.block(0, 0, rows, 3);
        Map(intensities.vector()) = mapped.block(0, 3, rows, 1);
        Map(colors.vector()) = mapped.block(0, 4, rows, 3);
        return true;
    }

    bool ReadXyz(const std::string &filepath, PointCloudInterface &pci) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<float> numbers = ParseNumbers(txt, cols, "LH");
        rows = numbers.size() / cols;

        auto mapped = Map(numbers, rows, cols);

        auto positions = pci.vertices.vertex_property<PointType>("v:point", PointType(0.0f));
        pci.vertices.resize(cols);
        Map(positions.vector()) = mapped.block(0, 0, 3, cols);

        if (cols == 6) {
            auto colors = pci.vertices.vertex_property<ColorType>("v:color", ColorType(0.0f));
            Map(colors.vector()) = mapped.block(3, 0, 3, cols);
        }
        return true;
    }


    bool ReadPly(const std::string &filepath, PointCloudInterface &pci) {
        happly::PLYData plyIn(filepath);

        // Get the vertex element
        happly::Element& vertexElement = plyIn.getElement("vertex");

        // Get the number of vertices
        size_t vertexCount = vertexElement.count;
        pci.vertices.resize(vertexCount);

        // Get positional data
        std::vector<float> x = vertexElement.getProperty<float>("x");
        std::vector<float> y = vertexElement.getProperty<float>("y");
        std::vector<float> z = vertexElement.getProperty<float>("z");

        // Get normal data
        std::vector<float> nx = vertexElement.getProperty<float>("nx");
        std::vector<float> ny = vertexElement.getProperty<float>("ny");
        std::vector<float> nz = vertexElement.getProperty<float>("nz");

        // Get the first three spherical harmonic coefficients
        std::vector<float> f_dc_0 = vertexElement.getProperty<float>("f_dc_0");
        std::vector<float> f_dc_1 = vertexElement.getProperty<float>("f_dc_1");
        std::vector<float> f_dc_2 = vertexElement.getProperty<float>("f_dc_2");

        // Get the remaining spherical harmonic coefficients
        std::vector<std::vector<float>> f_rest(45);
        for (int i = 0; i < 45; ++i) {
            f_rest[i] = vertexElement.getProperty<float>("f_rest_" + std::to_string(i));
        }

        // Get opacity
        std::vector<float> opacity = vertexElement.getProperty<float>("opacity");

        // Get scale data
        std::vector<float> scale_0 = vertexElement.getProperty<float>("scale_0");
        std::vector<float> scale_1 = vertexElement.getProperty<float>("scale_1");
        std::vector<float> scale_2 = vertexElement.getProperty<float>("scale_2");

        // Get rotation data
        std::vector<float> rot_0 = vertexElement.getProperty<float>("rot_0");
        std::vector<float> rot_1 = vertexElement.getProperty<float>("rot_1");
        std::vector<float> rot_2 = vertexElement.getProperty<float>("rot_2");
        std::vector<float> rot_3 = vertexElement.getProperty<float>("rot_3");

        // Get flag and curvature data
        std::vector<float> flag = vertexElement.getProperty<float>("flag");
        std::vector<float> curvature = vertexElement.getProperty<float>("curvature");


        auto position = pci.vertices.vertex_property<PointType>("v:point", PointType(0.0f));
        auto normal = pci.vertices.vertex_property<NormalType>("v:normal", NormalType(0.0f));
        auto spherical_harmonics = pci.vertices.vertex_property<Eigen::Vector<float, 48>>("v:spherical_harmonics", Eigen::Vector<float, 48>::Zero());
        auto op = pci.vertices.vertex_property<ScalarType>("v:opacity", 0.0f);
        auto scale = pci.vertices.vertex_property<Vector<float, 3>>("v:scale", Vector<float, 3>(0.0f));
        auto rotation = pci.vertices.vertex_property<Vector<float, 4>>("v:rotation", Vector<float, 4>(0.0f));
        auto axis = pci.vertices.vertex_property<Vector<float, 3>>("v:axis", Vector<float, 3>(0.0f));
        auto axis_norm = pci.vertices.vertex_property<float>("v:axis_norm", 0.0f);
        auto angle = pci.vertices.vertex_property<float>("v:angle", 0.0f);
        auto fl = pci.vertices.vertex_property<ScalarType>("v:flag", 0.0f);
        auto curv = pci.vertices.vertex_property<ScalarType>("v:curvature", 0.0f);
        // Combine the retrieved data into the vector of VertexData structs
        for (const auto &v : pci.vertices) {
            const auto i = v.idx();
            position[v] = {x[i], y[i], z[i]};
            normal[v] = {nx[i], ny[i], nz[i]};
            spherical_harmonics[v][0] = f_dc_0[i];
            spherical_harmonics[v][1] = f_dc_1[i];
            spherical_harmonics[v][2] = f_dc_2[i];
            for (int j = 3; j < 45; ++j) {
                spherical_harmonics[v][j] = f_rest[j][i];
            }
            op[v] = opacity[i];
            scale[v] = {scale_0[i], scale_1[i], scale_2[i]};
            rotation[v] = {rot_0[i], rot_1[i], rot_2[i], rot_3[i]};
            axis[v] = {rot_1[i], rot_2[i], rot_3[i]};
            axis_norm[v] = glm::length(axis[v]);
            angle[v] = rot_0[i];
            fl[v] = flag[i];
            curv[v] = curvature[i];
        }

        return true;
    }

    bool Write(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), tolower);

        if (ext == ".csv") {
            return WriteCsv(filepath, pci, flags);
        }
        if (ext == ".pts") {
            return WritePts(filepath, pci, flags);
        }
        if (ext == ".xyz") {
            return WriteXyz(filepath, pci, flags);
        }
        return false;
    }

    bool WriteCsv(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pci.vertices.get_vertex_property<PointType>("v:point");
        auto intensities = pci.vertices.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pci.vertices.get_vertex_property<ColorType>("v:color");
            for (const auto v: pci.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << " " << Map(colors[v]).transpose() <<
                        std::endl;
            }
        } else {
            for (const auto v: pci.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << std::endl;
            }
        }
        return true;
    }

    bool WritePts(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pci.vertices.get_vertex_property<PointType>("v:point");
        auto intensities = pci.vertices.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pci.vertices.get_vertex_property<ColorType>("v:color");
            for (const auto v: pci.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << " " << Map(colors[v]).transpose() <<
                        std::endl;
            }
        } else {
            for (const auto v: pci.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << std::endl;
            }
        }
        return true;
    }

    bool WriteXyz(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pci.vertices.get_vertex_property<PointType>("v:point");
        auto intensities = pci.vertices.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pci.vertices.get_vertex_property<ColorType>("v:color");
            for (const auto v: pci.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << " " << Map(colors[v]).transpose() <<
                        std::endl;
            }
        } else {
            for (const auto v: pci.vertices) {
                out << Map(positions[v]).transpose() << " " << intensities[v] << std::endl;
            }
        }
        return true;
    }
}
