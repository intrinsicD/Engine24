//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_COMMANDBUFFER_H
#define ENGINE24_COMMANDBUFFER_H

#include "Command.h"

namespace Bcg {
    struct CommandBuffer {
        template<class Command>
        CommandBuffer &add_command(Command &&command) {
            return add_command_sptr(std::make_shared<Command>(std::forward<Command>(command)));
        }

        CommandBuffer &add_command_sptr(std::shared_ptr<Commands::AbstractCommand> command);

        void execute() const;

        void clear();

    private:
        std::vector<std::shared_ptr<Commands::AbstractCommand>> commands;
    };
}

#endif //ENGINE24_COMMANDBUFFER_H
