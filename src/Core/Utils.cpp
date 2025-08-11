/*
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

    std::vector<float> ParseNumbers(const std::string &s, unsigned int &num_lines, const char *skip_chars) {
        const char *start = s.data();
        const char *end = s.data() + s.size();
        std::vector<float> numbers;
        num_lines = 0;
        while (start < end) {
            bool skip_line = false;
            while (start < end && (!std::isdigit(*start) && *start != '-' && *start != '.' && *start != '\n' && *start != '\t')) {
                if (skip_chars && strchr(skip_chars, *start)) {
                    skip_line = true;
                    while (start < end && *start != '\n') {
                        ++start;
                    }
                }
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
}*/

#include "Utils.h"
#include "fast_float/fast_float.h"
#include <fstream>
#include <cctype> // Required for isspace

namespace Bcg {
    std::string ReadTextFile(const std::string &filename) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file) {
            return "";
        }
        std::string txt;
        file.seekg(0, std::ios::end);
        txt.reserve(file.tellg());
        file.seekg(0, std::ios::beg);
        txt.assign((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
        return txt;
    }

    std::vector<float> ParseNumbers(const std::string &s, unsigned int &num_lines, const char *skip_chars,  const char *delimiters) {
        const char *start = s.data();
        const char *end = s.data() + s.size();
        std::vector<float> numbers;
        num_lines = 0;

        bool found_number_on_line = false;

        while (start < end) {
            // --- 1. Skip Delimiters and Whitespace ---
            // This advances 'start' past any character found in the 'delimiters' string.
            while (start < end && strchr(delimiters, *start)) {
                ++start;
            }

            // After skipping, if we are at the end, break the loop.
            if (start == end) break;

            // --- 2. Handle Newlines and Line Counting ---
            if (*start == '\n') {
                if (found_number_on_line) {
                    num_lines++;
                }
                found_number_on_line = false; // Reset for the next line
                ++start;
                continue; // Continue to the next loop iteration
            }

            // --- 3. Handle Comment Lines ---
            // If we find a comment character, skip until the next newline.
            if (skip_chars && strchr(skip_chars, *start)) {
                while (start < end && *start != '\n') {
                    ++start;
                }
                // Let the next loop iteration handle the newline character.
                continue;
            }

            // --- 4. Attempt to Parse a Number ---
            float value;
            auto result = fast_float::from_chars(start, end, value);
            if (result.ec == std::errc()) {
                // Success! We found a number.
                numbers.push_back(value);
                found_number_on_line = true; // Mark that this line is not empty.
                start = result.ptr;          // Move the pointer to after the parsed number.
            } else {
                // If it's not a number, not a delimiter, and not a newline,
                // it's an unexpected character. Skip it to prevent an infinite loop.
                ++start;
            }
        }

        // --- 5. Final Check for Last Line ---
        // If the file ends without a newline, we need to count the last line
        // if it contained any numbers.
        if (found_number_on_line) {
            num_lines++;
        }

        return numbers;
    }
}