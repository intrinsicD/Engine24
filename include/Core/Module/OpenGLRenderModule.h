//
// Created by alex on 11/24/24.
//

#ifndef OPENGLRENDERMODULE_H
#define OPENGLRENDERMODULE_H

#include "Module/Module.h"
#include "EventsMain.h"
#include "entt/signal/dispatcher.hpp"

namespace Bcg {
    class OpenGLRenderModule : public Module {
    public:
        //connects the event callbacks to the dispatcher on construction
        explicit OpenGLRenderModule(entt::dispatcher &dispatcher) : Module("OpenGLRenderModule"), dispatcher(dispatcher) {
            dispatcher.sink<Events::Initialize>().connect<&OpenGLRenderModule::on_initialize>(*this);
            dispatcher.sink<Events::Startup>().connect<&OpenGLRenderModule::on_startup>(*this);
            dispatcher.sink<Events::Synchronize>().connect<&OpenGLRenderModule::on_synchronize>(*this);
            dispatcher.sink<Events::Shutdown>().connect<&OpenGLRenderModule::on_shurdown>(*this);
        }

        //disconnects the event callbacks from the dispatcher on destruction
        ~OpenGLRenderModule() override{
            dispatcher.sink<Events::Initialize>().disconnect<&OpenGLRenderModule::on_initialize>(*this);
            dispatcher.sink<Events::Startup>().disconnect<&OpenGLRenderModule::on_startup>(*this);
            dispatcher.sink<Events::Synchronize>().disconnect<&OpenGLRenderModule::on_synchronize>(*this);
            dispatcher.sink<Events::Shutdown>().disconnect<&OpenGLRenderModule::on_shurdown>(*this);
        }

        void on_initialize(const Events::Initialize &event);

        void on_startup(const Events::Startup &event);

        void on_synchronize(const Events::Synchronize &event);

        void on_shurdown(const Events::Shutdown &event);

    private:
        entt::dispatcher &dispatcher;
    };
}

#endif //OPENGLRENDERMODULE_H
