//
// Created by alex on 04.07.24.
//

#include "GuiUtils.h"
#include "PropertiesGui.h"
#include "imgui.h"

namespace Bcg::Gui {
    bool Combo(const char *combo_label, std::pair<int, std::string> &curr, BasePropertyArray &a_property) {
        // Begin the combo box
        bool changed = false;
        if (ImGui::BeginCombo(combo_label, a_property.element_string(curr.first).c_str())) {
            ImGuiListClipper clipper;
            clipper.Begin(a_property.size(), ImGui::GetTextLineHeightWithSpacing());
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    bool is_selected = (curr.first == i);
                    if (ImGui::Selectable(a_property.element_string(i).c_str(), is_selected)) {
                        curr.first = i;
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }
            clipper.End();
            ImGui::EndCombo();
        }
        return changed;
    }

    bool ListBox(const char *listbox_label, std::pair<int, std::string> &curr, BasePropertyArray &a_property) {
        // Begin the list box
        bool changed = false;
        if (ImGui::BeginListBox(listbox_label, ImVec2(-FLT_MIN, 3 * ImGui::GetTextLineHeightWithSpacing()))) {
            ImGuiListClipper clipper;
            clipper.Begin(a_property.size(), ImGui::GetTextLineHeightWithSpacing());
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    bool is_selected = (curr.first == i);
                    if (ImGui::Selectable(a_property.element_string(i).c_str(), is_selected)) {
                        curr.first = i;
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }
            clipper.End();
            ImGui::EndListBox();
        }
        return changed;
    }

    bool Combo(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container) {
        auto property_names = container.properties();
        auto result = Combo(label, curr, property_names);
        curr.second = property_names[curr.first];
        return result;
    }

    bool ListBox(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container) {
        auto property_names = container.properties();
        auto result = ListBox(label, curr, property_names);
        curr.second = property_names[curr.first];
        return result;
    }

    void Show(BasePropertyArray &a_property) {
        static std::pair<int, std::string> curr = {0, ""};
        ImGui::PushID((a_property.name() + curr.second).c_str());
        Combo((curr.second + "##values").c_str(), curr, a_property);
        ImGui::PopID();
    }

    void Show(const char *label, PropertyContainer &container) {
        static std::pair<int, std::string> curr_property = {0, ""};
        Combo(label, curr_property, container);
        Show(*container.get_parray()[curr_property.first]);
        /*for (auto &a_property: container.get_parray()) {
            if (ImGui::TreeNode(a_property->name().c_str())) {
                ImGui::PushID(a_property->name().c_str());
                Show(*a_property);
                ImGui::PopID();
                ImGui::TreePop();
            }
        }*/
    }
}