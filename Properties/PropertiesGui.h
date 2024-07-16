//
// Created by alex on 04.07.24.
//

#ifndef ENGINE24_PROPERTIESGUI_H
#define ENGINE24_PROPERTIESGUI_H

#include "Properties.h"
#include "MatVec.h"

namespace Bcg::Gui {
    bool Combo(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container);

    bool ListBox(const char *label, std::pair<int, std::string> &curr, PropertyContainer &container);

    template<typename T>
    bool Combo(const char *label, std::pair<int, std::string> &curr, Property<T> &property) {
        return Combo(label, curr, ToStrings(property));
    }

    template<typename T>
    bool ListBox(const char *label, std::pair<int, std::string> &curr, Property<T> &property) {
        return ListBox(label, curr, ToStrings(property));
    }

    void Show(BasePropertyArray &basePropertyArray);
}

#endif //ENGINE24_PROPERTIESGUI_H
