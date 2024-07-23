//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_COMMANDBUFFER_H
#define ENGINE24_COMMANDBUFFER_H

#include "Command.h"
#include <cstring>

namespace Bcg {

    struct CommandDataBuffer {
        template<class Command>
        CommandDataBuffer &add_command(Command &&command) {
            size_t commandSize = command.size();
            size_t currentOffset = data.size();

            if (currentOffset + commandSize > capacity) {
                reserve((capacity == 0 ? 1 : capacity) * 2);
            }

            data.resize(currentOffset + commandSize);
            std::memcpy(data.data() + currentOffset, (void*)&command, commandSize);
            offsets.push_back(currentOffset);
            return *this;
        }

        void execute() const {
            for (size_t offset: offsets) {
                reinterpret_cast<const AbstractCommand *>(data.data() + offset)->execute();
            }
        }

        void clear() {
            data.clear();
            offsets.clear();
        }

        void reserve(size_t new_capacity) {
            if (new_capacity > capacity) {
                data.reserve(new_capacity);
                capacity = new_capacity;
            }
        }

    private:
        std::vector<uint8_t> data;
        std::vector<size_t> offsets;
        size_t capacity = 0;
    };

    struct CommandBuffer {
        template<class Command>
        CommandBuffer &add_command(Command &&command) {
            commands.push_back(std::make_shared<Command>(std::forward<Command>(command)));
            return *this;
        }

        CommandBuffer &add_command(std::shared_ptr<AbstractCommand> command) {
            commands.push_back(std::move(command));
            return *this;
        }

        void execute() const {
            for (const auto &cmd: commands) {
                cmd->execute();
            }
        }

        void clear() {
            commands.clear();
        }

    private:
        std::vector<std::shared_ptr<AbstractCommand>> commands;
    };

    struct DoubleCommandBuffer {
        CommandBuffer &current() {
            return *p_current;
        }

        CommandBuffer &next() {
            return *p_next;
        }

        void swap_buffers() {
            std::swap(p_current, p_next);
        }

    private:
        CommandBuffer a, b;

        CommandBuffer *p_current = &a;
        CommandBuffer *p_next = &b;
    };
}

#endif //ENGINE24_COMMANDBUFFER_H
