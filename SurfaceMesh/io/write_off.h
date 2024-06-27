// Copyright 2011-2022 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <filesystem>

#include "io_flags.h"
#include "../SurfaceMesh.h"

namespace Bcg {

void write_off(const SurfaceMesh& mesh, const std::filesystem::path& file,
               const IOFlags& flags);

} // namespace pmp