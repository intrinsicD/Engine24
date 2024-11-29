//
// Created by alex on 29.11.24.
//

#include "FileInputModule.h"
#include "Engine.h"

namespace Bcg{
    FileInputModule::FileInputModule() : Module("FileInputModule") {}

    void FileInputModule::activate() {
        if(base_activate()){
            Engine::Dispatcher().sink<Events::Callback::Drop>().connect<FileInputModule::on_drop>();
        }
    }

    void FileInputModule::deactivate() {
        if(base_deactivate()){
            Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<FileInputModule::on_drop>();
        }
    }

    void FileInputModule::on_drop(const Events::Callback::Drop &drop){
        //determine the file extension

        //forward to the appropriate handler
    }
}