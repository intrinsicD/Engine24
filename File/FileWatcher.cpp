//
// Created by alex on 20.06.24.
//

#include "FileWatcher.h"
#include <filesystem>

namespace Bcg {
    void FileWatcher::watch(File &file) {
        file.last_write_time = std::filesystem::last_write_time(file.path);
        watched[file.path] = file;
    }

    void FileWatcher::remove(const std::string &filepath) {
        watched.erase(filepath.c_str());
    }

    std::vector<File> FileWatcher::check(bool force) {
        std::vector<File> result;
        for (auto &item: watched) {
            auto &file = item.second;
            std::chrono::file_clock::time_point current_write_time = std::filesystem::last_write_time(file.path);
            if (force || file.last_write_time < current_write_time) {
                file.last_write_time = current_write_time;
                file.callback();
                result.emplace_back(file);
            }
        }
        return result;
    }
}