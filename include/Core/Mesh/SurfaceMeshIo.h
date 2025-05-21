//
// Created by alex on 12.08.24.
//

#ifndef ENGINE24_SURFACEMESHIO_H
#define ENGINE24_SURFACEMESHIO_H

#include "SurfaceMesh.h"
#include "MeshInterface.h"
#include "GeometryData.h"


namespace Bcg{
    struct IOFlags{
        bool use_binary = false;             //!< Read / write binary format.
        bool use_vertex_normals = false;     //!< Read / write vertex normals.
        bool use_vertex_colors = false;      //!< Read / write vertex colors.
        bool use_vertex_texcoords = false;   //!< Read / write vertex texcoords.
        bool use_face_normals = false;       //!< Read / write face normals.
        bool use_face_colors = false;        //!< Read / write face colors.
        bool use_halfedge_texcoords = false; //!< Read / write halfedge texcoords.
    };

    bool Read(const std::string &filepath, SurfaceMesh &mesh);

    bool ReadObj(const std::string &filepath, SurfaceMesh &mesh);

    bool ReadOff(const std::string &filepath, SurfaceMesh &mesh);

    bool ReadPly(const std::string &filepath, SurfaceMesh &mesh);

    bool ReadStl(const std::string &filepath, SurfaceMesh &mesh);

    bool ReadPmp(const std::string &filepath, SurfaceMesh &mesh);

    bool Write(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags = IOFlags());

    bool WriteObj(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags = IOFlags());

    bool WriteOff(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags = IOFlags());

    bool WritePly(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags = IOFlags());

    bool WriteStl(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags = IOFlags());

    bool WritePmp(const std::string &filepath, const SurfaceMesh &mesh, const IOFlags &flags = IOFlags());

    bool Read(const std::string &filepath, HalfedgeMeshInterface &mesh);

    bool ReadObj(const std::string &filepath, HalfedgeMeshInterface &mesh);

    bool ReadOff(const std::string &filepath, HalfedgeMeshInterface &mesh);

    bool ReadPly(const std::string &filepath, HalfedgeMeshInterface &mesh);

    bool ReadStl(const std::string &filepath, HalfedgeMeshInterface &mesh);

    bool ReadPmp(const std::string &filepath, HalfedgeMeshInterface &mesh);

    bool Write(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags = IOFlags());

    bool WriteObj(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags = IOFlags());

    bool WriteOff(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags = IOFlags());

    bool WritePly(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags = IOFlags());

    bool WriteStl(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags = IOFlags());

    bool WritePmp(const std::string &filepath, const HalfedgeMeshInterface &mesh, const IOFlags &flags = IOFlags());
}

#endif //ENGINE24_SURFACEMESHIO_H
