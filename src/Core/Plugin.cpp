//
// Created by alex on 19.06.24.
//

#include "Plugin.h"
#include "Logger.h"
#include <string>

namespace Bcg {
    Plugin::Plugin(const char *name) : name(name) {}

    void Plugin::activate() {
        Log::Info(("Activate " + std::string(name)).c_str());
    }

    void Plugin::deactivate() {
        Log::Info(("Deactivate " + std::string(name)).c_str());
    }
}