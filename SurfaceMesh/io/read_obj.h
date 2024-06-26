// Copyright 2011-2022 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <filesystem>

#include "../SurfaceMesh.h"

namespace Bcg {

void read_obj(SurfaceMesh& mesh, const std::filesystem::path& file);

} // namespace pmp