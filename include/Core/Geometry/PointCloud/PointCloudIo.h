//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_POINTCLOUDIO_H
#define ENGINE24_POINTCLOUDIO_H

#include "PointCloud.h"
#include "PointCloudInterface.h"
#include "GeometryData.h"

namespace Bcg{

    struct PointCloudIOFlags {
        bool use_binary = false;             //!< Read / write binary format.
        bool use_normals = false;     //!< Read / write vertex normals.
        bool use_colors = false;      //!< Read / write vertex colors.
    };

    bool Read(const std::string &filepath, PointCloud &pc);

    bool ReadCsv(const std::string &filepath, PointCloud &pc);

    bool ReadPts(const std::string &filepath, PointCloud &pc);

    bool ReadXyz(const std::string &filepath, PointCloud &pc);

    bool ReadPly(const std::string &filepath, PointCloud &pc);

    bool Write(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags = PointCloudIOFlags());

    bool WriteCsv(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags = PointCloudIOFlags());

    bool WritePts(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags = PointCloudIOFlags());

    bool WriteXyz(const std::string &filepath, const PointCloud &pc, const PointCloudIOFlags &flags = PointCloudIOFlags());


    bool Read(const std::string &filepath, PointCloudInterface &pci);

    bool ReadCsv(const std::string &filepath, PointCloudInterface &pci);

    bool ReadPts(const std::string &filepath, PointCloudInterface &pci);

    bool ReadXyz(const std::string &filepath, PointCloudInterface &pci);

    bool ReadPly(const std::string &filepath, PointCloudInterface &pci);

    bool Write(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags = PointCloudIOFlags());

    bool WriteCsv(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags = PointCloudIOFlags());

    bool WritePts(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags = PointCloudIOFlags());

    bool WriteXyz(const std::string &filepath, const PointCloudInterface &pci, const PointCloudIOFlags &flags = PointCloudIOFlags());
}

#endif //ENGINE24_POINTCLOUDIO_H
