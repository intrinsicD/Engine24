//
// Created by alex on 04.07.24.
//

#include "GuiUtils.h"
#include "imgui.h"

namespace Bcg::Gui {
    bool Combo(const char *combo_label, std::pair<int, std::string> &curr, std::vector<std::string> &labels) {
        bool changed = false;
        if (ImGui::BeginCombo(combo_label, labels[curr.first].c_str())) {
            ImGuiListClipper clipper;
            clipper.Begin(labels.size(), ImGui::GetTextLineHeightWithSpacing());
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    bool is_selected = (curr.first == i);
                    if (ImGui::Selectable(labels[i].c_str(), is_selected)) {
                        curr.first = i;
                        curr.second = labels[i];
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


    bool ListBox(const char *listbox_label, std::pair<int, std::string> &curr, std::vector<std::string> &labels) {
        bool changed = false;
        if (ImGui::BeginListBox(listbox_label, ImVec2(-FLT_MIN, 3 * ImGui::GetTextLineHeightWithSpacing()))) {
            ImGuiListClipper clipper;
            clipper.Begin(labels.size(), ImGui::GetTextLineHeightWithSpacing());
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    bool is_selected = (curr.first == i);
                    if (ImGui::Selectable(labels[i].c_str(), is_selected)) {
                        curr.first = i;
                        curr.second = labels[i];
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

    int FindIndex(const std::vector<std::string> &labels, std::string label) {
        for (int i = 0; i < labels.size(); ++i) {
            if (labels[i] == label) return i;
        }
        return -1;
    }

}