//
// Created by alex on 12.08.24.
//

#include "CommandBuffer.h"

namespace Bcg {
    CommandBuffer &CommandBuffer::add_command_sptr(std::shared_ptr<Commands::AbstractCommand> command) {
        commands.push_back(std::move(command));
        return *this;
    }

    void CommandBuffer::execute() const {
        for (const auto &cmd: commands) {
            cmd->execute();
        }
    }

    void CommandBuffer::clear() {
        commands.clear();
    }
}