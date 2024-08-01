//
// Created by alex on 20.06.24.
//

#ifndef ENGINE24_FILEWATCHER_H
#define ENGINE24_FILEWATCHER_H

#include <unordered_map>
#include <functional>
#include <chrono>

namespace Bcg {
    struct File {
        std::string path;
        std::function<void()> callback;
        std::chrono::file_clock::time_point last_write_time;
    };

    class FileWatcher {
    public:
        FileWatcher() = default;

        void watch(File &file);

        void remove(const std::string &filepath);

        std::vector<File> check(bool force = false);

        std::unordered_map<std::string, File> watched;
    };
}

#endif //ENGINE24_FILEWATCHER_H
