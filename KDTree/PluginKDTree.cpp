//
// Created by alex on 01.08.24.
//

#include "PluginKDTree.h"
#include "Engine.h"
#include "imgui.h"
#include "Logger.h"
#include "KDTreeCpu.h"
#include "Picker.h"

namespace Bcg {
    void PluginKDTree::activate() {
        Plugin::activate();
    }

    void PluginKDTree::begin_frame() {}

    void PluginKDTree::update() {}

    void PluginKDTree::end_frame() {}

    void PluginKDTree::deactivate() {
        Plugin::deactivate();
    }

    void PluginKDTree::render_menu() {}

    void PluginKDTree::render_gui() {}

    void PluginKDTree::render() {}
}