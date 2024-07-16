//
// Created by alex on 04.07.24.
//

#ifndef ENGINE24_GUIUTILS_H
#define ENGINE24_GUIUTILS_H

#include <sstream>
#include <vector>

namespace Bcg::Gui {
    bool Combo(const char *label, std::pair<int, std::string> &curr, std::vector<std::string> &labels);

    bool ListBox(const char *label, std::pair<int, std::string> &curr, std::vector<std::string> &labels);

    template<typename Iterable>
    std::vector<std::string> ToStrings(const Iterable &iterable) {
        std::vector<std::string> values;
        values.reserve(std::distance(iterable.begin(), iterable.end()));

        std::stringstream ss;
        for (const auto &item: iterable) {
            ss.str("");
            ss.clear();
            ss << item;
            values.emplace_back(ss.str());
        }

        return values;
    }
}

#endif //ENGINE24_GUIUTILS_H
