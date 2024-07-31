//
// Created by alex on 30.07.24.
//

#include "read_xyz.h"
#include "Utils.h"
#include "PropertyEigenMap.h"

namespace Bcg {
    void read_xyz(PointCloud &pc, const std::string &filename) {
        auto txt = ReadTextFile(filename);
        unsigned int rows, cols;
        std::vector<float> numbers = ParseNumbers(txt, cols, "LH");
        rows = numbers.size() / cols;

        auto mapped = Map<float, 3, -1>(numbers, rows, cols);

        pc.vprops_.resize(cols);
        Map(pc.positions()) = mapped.block(0, 0, 3, cols);

        if (cols == 6) {
            auto colors = pc.vertex_property<Color>("v:color", Color::Zero());
            Map(colors.vector()) = mapped.block(3, 0, 3, cols);
        }
    }
}