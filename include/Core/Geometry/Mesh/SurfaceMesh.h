// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once


#include "MeshInterface.h"
#include "GeometryData.h"

namespace Bcg {
    struct IOFlags;

    class SurfaceMesh {
    public:
        SurfaceMesh() : interface(data) {

        }

        virtual ~SurfaceMesh() = default;

        SurfaceMesh(const SurfaceMesh &other) : data(other.data),
                                                interface(data) {
        }

        SurfaceMesh &operator=(const SurfaceMesh &other) {
            if (this != &other) {
                data = other.data;
                interface = HalfedgeMeshInterface(data);
            }
            return *this;
        }

        // Define move constructor
        SurfaceMesh(SurfaceMesh &&other) noexcept
            : data(std::move(other.data)),
              interface(data) {
        }

        // Define move assignment operator
        SurfaceMesh &operator=(SurfaceMesh &&other) noexcept {
            if (this != &other) {
                data = std::move(other.data);
                interface = HalfedgeMeshInterface(data);
            }
            return *this;
        }

        MeshData data;
        HalfedgeMeshInterface interface;
    };

    //!@}
} // namespace pmp
