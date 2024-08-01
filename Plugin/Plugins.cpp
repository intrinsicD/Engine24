//
// Created by alex on 27.06.24.
//

#include "Plugins.h"
#include "Input.h"
#include "Logger.h"
#include "PluginAABB.h"
#include "PluginMesh.h"
#include "PluginPointCloud.h"
#include "PluginPicker.h"
#include "PluginFrameTimer.h"
#include "PluginCamera.h"
#include "PluginTransform.h"
#include "PluginHierarchy.h"
#include "PluginViews.h"
#include <unordered_map>

namespace Bcg {
    static std::unordered_map<std::string, std::unique_ptr<Plugin>> plugins;

    void Plugins::init() {
        add_plugin(std::make_unique<Input>());
        add_plugin(std::make_unique<PluginAABB>());
        add_plugin(std::make_unique<PluginMesh>());
        add_plugin(std::make_unique<PluginPointCloud>());
        add_plugin(std::make_unique<PluginPicker>());
        add_plugin(std::make_unique<PluginCamera>());
        add_plugin(std::make_unique<PluginFrameTimer>());
        add_plugin(std::make_unique<PluginTransform>());
        add_plugin(std::make_unique<PluginHierarchy>());
        add_plugin(std::make_unique<PluginViews>());
    }

    void Plugins::init_user_plugin(const std::string &name) {
        Log::TODO("implement user dynlib loading");
    }

    void Plugins::add_plugin(std::unique_ptr<Plugin> uptr) {
        plugins[uptr->name] = std::forward<std::unique_ptr<Plugin>>(uptr);
    }

    void Plugins::activate_all() {
        for (auto &[name, plugin]: plugins) {
            plugin->activate();
        }
    }

    void Plugins::begin_frame_all() {
        for (auto &[name, plugin]: plugins) {
            plugin->begin_frame();
        }
    }

    void Plugins::update_all() {
        for (auto &[name, plugin]: plugins) {
            plugin->update();
        }
    }

    void Plugins::render_all() {
        for (auto &[name, plugin]: plugins) {
            plugin->render();
        }
    }

    void Plugins::render_menu() {
        for (auto &[name, plugin]: plugins) {
            plugin->render_menu();
        }
    }

    void Plugins::render_gui() {
        for (auto &[name, plugin]: plugins) {
            plugin->render_gui();
        }
    }

    void Plugins::end_frame() {
        for (auto &[name, plugin]: plugins) {
            plugin->end_frame();
        }
    }

    void Plugins::deactivate_all() {
        for (auto &[name, plugin]: plugins) {
            plugin->deactivate();
        }
    }
}