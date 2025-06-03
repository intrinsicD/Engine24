//
// Created by alex on 03.06.25.
//

#ifndef ENGINE24_STRINGUTILS_H
#define ENGINE24_STRINGUTILS_H

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>

namespace Bcg::StringUtils {
    // Converts a string to lowercase
    inline std::string to_lower(const std::string &str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }

    // Converts a string to uppercase
    inline std::string to_upper(const std::string &str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
    }

    // Trims whitespace from both ends of a string
    inline std::string trim(const std::string &str) {
        const auto start = str.find_first_not_of(" \t\n\r");
        const auto end = str.find_last_not_of(" \t\n\r");
        return (start == std::string::npos || end == std::string::npos) ? "" : str.substr(start, end - start + 1);
    }

    // Splits a string by a delimiter and returns a vector of substrings
    inline std::vector<std::string> split(const std::string &str, const std::string &delimiter) {
        std::vector<std::string> tokens;

        size_t start = 0;
        size_t end = str.find(delimiter);
        while (end != std::string::npos) {
            tokens.push_back(str.substr(start, end - start));
            start = end + delimiter.length();
            end = str.find(delimiter, start);
        }
        tokens.push_back(str.substr(start));

        return tokens;
    }

    // Joins a vector of strings into a single string with a specified delimiter
    inline std::string join(const std::vector<std::string> &strings, const std::string &delimiter) {
        std::ostringstream oss;
        for (size_t i = 0; i < strings.size(); ++i) {
            oss << strings[i];
            if (i < strings.size() - 1) {
                oss << delimiter;
            }
        }
        return oss.str();
    }
}

#endif //ENGINE24_STRINGUTILS_H
