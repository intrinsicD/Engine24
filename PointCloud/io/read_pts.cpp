//
// Created by alex on 30.07.24.
//

#include "read_pts.h"
#include "Utils.h"
#include "PropertyEigenMap.h"

namespace Bcg {
    void read_pts(PointCloud &pc, const std::string &filename) {
        auto txt = ReadTextFile(filename);
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
    }
}