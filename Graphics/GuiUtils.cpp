//
// Created by alex on 04.07.24.
//

#include "GuiUtils.h"
#include "imgui.h"

namespace Bcg::Gui {
    static auto VectorGetter = [](void *vec, int idx, const char **out_text) {
        auto &vector = *static_cast<std::vector<std::string> *>(vec);
        if (idx < 0 || idx >= static_cast<int>(vector.size())) {
            return false;
        }
        *out_text = vector.at(idx).c_str();
        return true;
    };

    bool Combo(const char *label, std::pair<int, std::string> &curr, std::vector<std::string> &labels) {
        auto result = ImGui::Combo(label, &curr.first, VectorGetter, static_cast<void *>(&labels),
                                   static_cast<int>(labels.size()));
        if (result) {
            curr.second = labels[curr.first];
        }
        return result;
    }


    bool ListBox(const char *label, std::pair<int, std::string> &curr, std::vector<std::string> &labels) {
        auto result = ImGui::ListBox(label, &curr.first, VectorGetter, static_cast<void *>(&labels),
                                     static_cast<int>(labels.size()));
        if (result) {
            curr.second = labels[curr.first];
        }
        return result;
    }

}