//
// Created by alex on 03.07.24.
//

#ifndef ENGINE24_PICKER_H
#define ENGINE24_PICKER_H

#include "entt/fwd.hpp"
#include "../MatVec.h"
#include "Plugin.h"

namespace Bcg {
    struct Picked {
        struct Entity {
            entt::entity id;
            bool is_background = true;

            operator bool() { return !is_background; }

            unsigned int vertex_idx = -1;
            unsigned int edge_idx = -1;
            unsigned int face_idx = -1;
        } entity;
    };

    class Picker : public Plugin {
    public:
        Picker();

        ~Picker() override = default;

        static Picked &pick(double x, double y);

        static Picked &last_picked();
    };
}

#endif //ENGINE24_PICKER_H
