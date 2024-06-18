//
// Created by alex on 18.06.24.
//

#include "Logger.h"
#include <iostream>

namespace Bcg::Log {
    void Info(const char *message) {
        std::cout << "Info:  " << message << "\n";
    }

    void Warn(const char *message) {
        std::cout << "Warn:  " << message << "\n";
    }

    void Error(const char *message) {
        std::cerr << "Error: " << message << "\n";
    }
}