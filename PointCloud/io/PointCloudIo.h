//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_POINTCLOUDIO_H
#define ENGINE24_POINTCLOUDIO_H

#include "PointCloud.h"

#include <filesystem>

namespace Bcg {

    struct PointCloudIOFlags {
        bool use_binary = false;             //!< Read / write binary format.
        bool use_normals = false;     //!< Read / write vertex normals.
        bool use_colors = false;      //!< Read / write vertex colors.
    };

//! \brief Read into \p mesh from \p file
//! \details File extension determines file type. Supported formats and
//! vertex attributes (a=ASCII, b=binary):
//!
//! Format | ASCII | Binary | Normals | Colors | Texcoords
//! -------|-------|--------|---------|--------|----------
//! OBJ    | yes   | no     | a       | no     | no
//! OFF    | yes   | yes    | a / b   | a      | a / b
//! PMP    | no    | yes    | no      | no     | no
//! STL    | yes   | yes    | no      | no     | no
//!
//! In addition, the OBJ and PMP formats support reading per-halfedge
//! texture coordinates.
//! \ingroup io
    void read(PointCloud &mesh, const std::filesystem::path &file);

//! \brief Write \p mesh to \p file controlled by \p flags
//! \details File extension determines file type. Supported formats and
//! vertex attributes (a=ASCII, b=binary):
//!
//! Format | ASCII | Binary | Normals | Colors | Texcoords
//! -------|-------|--------|---------|--------|----------
//! OBJ    | yes   | no     | a       | no     | no
//! OFF    | yes   | yes    | a       | a      | a
//! PMP    | no    | yes    | no      | no     | no
//! STL    | yes   | yes    | no      | no     | no
//!
//! In addition, the OBJ and PMP formats support writing per-halfedge
//! texture coordinates.
//! \ingroup io
    void write(const PointCloud &mesh, const std::filesystem::path &file,
               const PointCloudIOFlags &flags = PointCloudIOFlags());

} // namespace pmp

#endif //ENGINE24_POINTCLOUDIO_H
