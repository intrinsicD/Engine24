//
// Created by alex on 18.06.24.
//

#include "Logger.h"
#include <iostream>
#include <string>

namespace Bcg::Log {
    std::string RED() {
        return "\033[1;31m";
    }

    std::string GREEN() {
        return "\033[1;32m";
    }

    std::string YELLOW() {
        return "\033[1;33m";
    }

    std::string WHITE() {
        return "\033[0m";
    }

    void Info(const std::string &message) {
        std::cout << GREEN() << "Info:  " << WHITE() << message << "\n";
    }

    void Warn(const std::string &message) {
        std::cout << YELLOW() << "Warn:  " << WHITE() << message << "\n";
    }

    void Error(const std::string &message) {
        std::cout << RED() << "Error: " << WHITE() << message << "\n";
    }

    void TODO(const std::string &message) {
        std::cout << RED() << "TODO:  " << YELLOW() << message << WHITE() << "\n";
    }

    void Progress(unsigned int iter, unsigned int size, unsigned int bar_width) {
        Progress(float(iter) / float(size-1), bar_width);
    }

    void Progress(float progress, unsigned int bar_width) {
        std::cout << "\r["; // Move to the beginning of the line
        size_t pos = bar_width * progress;
        for (size_t i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %";
        std::cout.flush();
        if (progress >= 1.0) {
            std::cout << std::endl;
        }
    }
}