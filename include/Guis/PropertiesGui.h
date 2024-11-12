//
// Created by alex on 04.07.24.
//

#ifndef ENGINE24_PROPERTIESGUI_H
#define ENGINE24_PROPERTIESGUI_H

#include "Properties.h"
#include "MatVec.h"

namespace Bcg::Gui {
    bool Combo(const char *label, std::pair<int, std::string> &curr, const BasePropertyArray &a_property);

    bool ListBox(const char *label, std::pair<int, std::string> &curr, const BasePropertyArray &a_property);

    bool Combo(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container);

    bool ListBox(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container);

    void Show(BasePropertyArray &basePropertyArray);

    void Show(const BasePropertyArray &basePropertyArray);

    void Show(const char *label, PropertyContainer &container);
}

#endif //ENGINE24_PROPERTIESGUI_H
