//
// Created by alex on 30.07.24.
//

#include "Utils.h"
#include "fast_float/fast_float.h"
#include <fstream>

namespace Bcg {
    std::string ReadTextFile(const std::string &filename) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        std::string txt;
        if (!file) {
            return txt;
        }

        // Get the file size and reserve the string capacity
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        txt.reserve(fileSize);
        file.seekg(0, std::ios::beg);

        // Use istreambuf_iterator to read the file contents into the string
        txt.assign((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
        return txt;
    }

    std::vector<float> ParseNumbers(const std::string &s, unsigned int &num_lines) {
        const char *start = s.data();
        const char *end = s.data() + s.size();
        std::vector<float> numbers;
        num_lines = 0;
        while (start < end) {
            while (start < end && (!std::isdigit(*start) && *start != '-' && *start != '.' && *start != '\n' && *start != '\t')) {
                ++start;
            }
            // Handle newline and tab characters separately
            if (start < end) {
                if (*start == '\n') {
                    ++num_lines;
                    ++start;
                    continue;
                } else if (*start == '\t') {
                    ++start;
                    continue;
                }
            }

            float value;
            auto result = fast_float::from_chars(start, end, value);
            if (result.ec == std::errc()) {
                numbers.push_back(value);
                start = result.ptr + 1;
            }
        }
        return numbers;
    }
}