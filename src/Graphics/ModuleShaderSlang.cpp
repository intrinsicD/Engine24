//
// Created by alex on 6/4/25.
//

#include "ModuleShaderSlang.h"
#include "Engine.h"
#include "Logger.h"
#include <slang.h>
#include <slang-com-helper.h>
#include <slang-com-ptr.h>



namespace Bcg {
    struct SlangData {
        Slang::ComPtr<slang::IGlobalSession> global_session;
        Slang::ComPtr<slang::ISession> compile_session;
        slang::SessionDesc session_desc = {};
        std::vector<slang::TargetDesc> targets;  // array of target descriptors

        // We also need a place to hold the linked program so that
        // Slang reflection or code generation can reference it later:
        Slang::ComPtr<slang::IModule>       module;
        Slang::ComPtr<slang::IComponentType> composite_component;
        Slang::ComPtr<slang::IComponentType> linked_program;
    };

    void ModuleShaderSlang::activate() {
        auto &slang = Engine::Context().emplace<SlangData>();
        if (!slang::createGlobalSession(slang.global_session.writeRef())) {
            Log::Error("{}: Failed to create global session", name);
            return;
        }
        //Whats missing?

    }

    void ModuleShaderSlang::deactivate() {
        if (base_deactivate()) {
            Slang::ComPtr<slang::IGlobalSession> global_session;
            Slang::ComPtr<slang::ISession> compile_session;
            slang::SessionDesc session_desc = {};
            std::vector<slang::TargetDesc> targets;  // array of target descriptors

            // We also need a place to hold the linked program so that
            // Slang reflection or code generation can reference it later:
            Slang::ComPtr<slang::IModule>       module;
            Slang::ComPtr<slang::IComponentType> composite_component;
            Slang::ComPtr<slang::IComponentType> linked_program;

            auto &slangData = Engine::Context().get<SlangData>();
        }
    }

    void ModuleShaderSlang::update() {
    }

    void ModuleShaderSlang::render_menu() {
    }

    void ModuleShaderSlang::render_gui() {
    }
}
