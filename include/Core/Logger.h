//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_LOGGER_H
#define ENGINE24_LOGGER_H

#include <fmt/core.h>
#include <fmt/color.h>

namespace Bcg::Log {
    void Info(const std::string &message);

    template<typename... Args>
    void Info(fmt::format_string<Args...> format, Args &&... args) {
        std::string message = fmt::format(format, std::forward<Args>(args)...);
        Info(message);
    }

    void Warn(const std::string &message);

    template<typename... Args>
    void Warn(fmt::format_string<Args...> format, Args &&... args) {
        std::string message = fmt::format(format, std::forward<Args>(args)...);
        Warn(message);
    }

    void Error(const std::string &message);

    template<typename... Args>
    void Error(fmt::format_string<Args...> format, Args &&... args) {
        std::string message = fmt::format(format, std::forward<Args>(args)...);
        Error(message);
    }

    void TODO(const std::string &message);

    template<typename... Args>
    void TODO(fmt::format_string<Args...> format, Args &&... args) {
        std::string message = fmt::format(format, std::forward<Args>(args)...);
        TODO(message);
    }

    void Progress(unsigned int iter, unsigned int size, unsigned int bar_width = 50);

    void Progress(float progress, unsigned int bar_width = 50);
}
#endif //ENGINE24_LOGGER_H
