//
// Created by alex on 18.06.24.
//

#include "Logger.h"
#include <iostream>
#include <string>

namespace Bcg::Log {
    void Info(const std::string &message) {
        std::cout << "Info:  " << message << "\n";
    }

    void Warn(const std::string &message) {
        std::cout << "Warn:  " << message << "\n";
    }

    void Error(const std::string &message) {
        std::cerr << "Error: " << message << "\n";
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