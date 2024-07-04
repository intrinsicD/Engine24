//
// Created by alex on 04.07.24.
//

#include "GuiUtils.h"
#include "PropertiesGui.h"
#include "imgui.h"

namespace Bcg::Gui {
    bool Combo(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container) {
        auto property_names = container.properties();
        auto result = Combo(label, curr, property_names);
        if (result) {
            curr.second = property_names[curr.first];
        }
        return result;
    }

    bool ListBox(const char *label, std::pair<int, std::string> &curr,
                                         PropertyContainer &container) {
        auto property_names = container.properties();
        auto result = ListBox(label, curr, property_names);
        if (result) {
            curr.second = property_names[curr.first];
        }
        return result;
    }

    void Show(BasePropertyArray &basePropertyArray) {
        ImGui::Text("name: %s", basePropertyArray.name().c_str());
    }
}