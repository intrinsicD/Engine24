//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_LOGGER_H
#define ENGINE24_LOGGER_H

#include <string>

namespace Bcg::Log {
    void Info(const std::string &message);

    void Warn(const std::string &message);

    void Error(const std::string &message);

    void Progress(unsigned int iter, unsigned int size, unsigned int bar_width = 50);

    void Progress(float progress, unsigned int bar_width = 50);
}
#endif //ENGINE24_LOGGER_H
