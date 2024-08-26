//
// Created by alex on 13.08.24.
//

#include "PointCloudIo.h"
#include "Utils.h"
#include "PropertyEigenMap.h"
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
        auto mapped = Map<ScalarType, -1, -1>(numbers, rows, cols);
        auto colors = pc.vertex_property<ColorType>("v:color", ColorType::Zero());
        auto intensities = pc.vertex_property<ScalarType>("v:intensity", 1);

        pc.vprops_.resize(rows);
        Map(pc.positions()) = mapped.block(0, 0, rows, 3);
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
        auto mapped = Map<ScalarType, -1, -1>(numbers, rows, cols);
        auto colors = pc.vertex_property<ColorType>("v:color", ColorType::Zero());
        auto intensities = pc.vertex_property<ScalarType>("v:intensity", 1);

        pc.vprops_.resize(rows);
        Map(pc.positions()) = mapped.block(0, 0, rows, 3);
        Map(intensities.vector()) = mapped.block(0, 3, rows, 1);
        Map(colors.vector()) = mapped.block(0, 4, rows, 3);
        return true;
    }

    bool ReadXyz(const std::string &filepath, PointCloud &pc) {
        auto txt = ReadTextFile(filepath);
        if (txt.empty()) return false;
        unsigned int rows, cols;
        std::vector<float> numbers = ParseNumbers(txt, cols, "LH");
        rows = numbers.size() / cols;

        auto mapped = Map<float, 3, -1>(numbers, rows, cols);

        pc.vprops_.resize(cols);
        Map(pc.positions()) = mapped.block(0, 0, 3, cols);

        if (cols == 6) {
            auto colors = pc.vertex_property<ColorType>("v:color", ColorType::Zero());
            Map(colors.vector()) = mapped.block(3, 0, 3, cols);
        }
        return true;
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

        auto positions = pc.get_vertex_property<PointType>("v:position");
        auto intensities = pc.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pc.get_vertex_property<ColorType>("v:color");
            for (const auto v: pc.vertices()) {
                out << positions[v].transpose() << " " << intensities[v] << " " << colors[v].transpose() << std::endl;
            }
        } else {
            for (const auto v: pc.vertices()) {
                out << positions[v].transpose() << " " << intensities[v] << std::endl;
            }
        }
        return false;
    }

    bool WritePts(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pc.get_vertex_property<PointType>("v:position");
        auto intensities = pc.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pc.get_vertex_property<ColorType>("v:color");
            for (const auto v: pc.vertices()) {
                out << positions[v].transpose() << " " << intensities[v] << " " << colors[v].transpose() << std::endl;
            }
        } else {
            for (const auto v: pc.vertices()) {
                out << positions[v].transpose() << " " << intensities[v] << std::endl;
            }
        }
        return false;
    }

    bool WriteXyz(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pc.get_vertex_property<PointType>("v:position");
        auto intensities = pc.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pc.get_vertex_property<ColorType>("v:color");
            for (const auto v: pc.vertices()) {
                out << positions[v].transpose() << " " << intensities[v] << " " << colors[v].transpose() << std::endl;
            }
        } else {
            for (const auto v: pc.vertices()) {
                out << positions[v].transpose() << " " << intensities[v] << std::endl;
            }
        }
        return false;
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
        auto mapped = Map<ScalarType, -1, -1>(numbers, rows, cols);

        auto positions = pci.vertices.vertex_property<PointType>("v:point", PointType::Zero());
        auto colors = pci.vertices.vertex_property<ColorType>("v:color", ColorType::Zero());
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
        auto mapped = Map<ScalarType, -1, -1>(numbers, rows, cols);

        auto positions = pci.vertices.vertex_property<PointType>("v:point", PointType::Zero());
        auto colors = pci.vertices.vertex_property<ColorType>("v:color", ColorType::Zero());
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

        auto mapped = Map<float, 3, -1>(numbers, rows, cols);

        auto positions = pci.vertices.vertex_property<PointType>("v:point", PointType::Zero());
        pci.vertices.resize(cols);
        Map(positions.vector()) = mapped.block(0, 0, 3, cols);

        if (cols == 6) {
            auto colors = pci.vertices.vertex_property<ColorType>("v:color", ColorType::Zero());
            Map(colors.vector()) = mapped.block(3, 0, 3, cols);
        }
        return true;
    }

    bool Write(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        auto ext = std::filesystem::path(filepath).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), tolower);

        if (ext == ".csv") {
            return WriteCsv(filepath, pci, flags);
        } else if (ext == ".pts") {
            return WritePts(filepath, pci, flags);
        } else if (ext == ".xyz") {
            return WriteXyz(filepath, pci, flags);
        } else {
            return false;
        }
    }

    bool WriteCsv(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pci.vertices.get_vertex_property<PointType>("v:position");
        auto intensities = pci.vertices.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pci.vertices.get_vertex_property<ColorType>("v:color");
            for (const auto v: pci.vertices) {
                out << positions[v].transpose() << " " << intensities[v] << " " << colors[v].transpose() << std::endl;
            }
        } else {
            for (const auto v: pci.vertices) {
                out << positions[v].transpose() << " " << intensities[v] << std::endl;
            }
        }
        return false;
    }

    bool WritePts(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pci.vertices.get_vertex_property<PointType>("v:position");
        auto intensities = pci.vertices.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pci.vertices.get_vertex_property<ColorType>("v:color");
            for (const auto v: pci.vertices) {
                out << positions[v].transpose() << " " << intensities[v] << " " << colors[v].transpose() << std::endl;
            }
        } else {
            for (const auto v: pci.vertices) {
                out << positions[v].transpose() << " " << intensities[v] << std::endl;
            }
        }
        return false;
    }

    bool WriteXyz(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags) {
        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        auto positions = pci.vertices.get_vertex_property<PointType>("v:position");
        auto intensities = pci.vertices.get_vertex_property<ScalarType>("v:intensity");

        if (flags.use_colors) {
            auto colors = pci.vertices.get_vertex_property<ColorType>("v:color");
            for (const auto v: pci.vertices) {
                out << positions[v].transpose() << " " << intensities[v] << " " << colors[v].transpose() << std::endl;
            }
        } else {
            for (const auto v: pci.vertices) {
                out << positions[v].transpose() << " " << intensities[v] << std::endl;
            }
        }
        return false;
    }
}