//
// Created by alex on 20.06.24.
//

#ifndef ENGINE24_COMMAND_H
#define ENGINE24_COMMAND_H

#include <functional>
#include <memory>

namespace Bcg {
    struct AbstractCommand {
        explicit AbstractCommand(const std::string &name) : name(name) {}

        virtual ~AbstractCommand() = default;

        void operator()() const { return execute(); }

        virtual void execute() const = 0;

        std::string name;
    };

    struct Task : public AbstractCommand {
        Task(const std::string &name, std::function<void()> callback) : AbstractCommand(name), callback(std::move(callback)) {}

        ~Task() override = default;

        void execute() const override {
            if (callback) {
                callback();
            }
        }

        std::function<void()> callback;
    };

    struct CompositeCommand : public AbstractCommand {
        explicit CompositeCommand(const std::string &name) : AbstractCommand(name) {}

        ~CompositeCommand() override = default;

        CompositeCommand &add_command(std::shared_ptr<AbstractCommand> sptr) {
            commands.emplace_back(sptr);
            return *this;
        }

        void execute() const override {
            for (auto cmd: commands) {
                cmd->execute();
            }
        }

    protected:
        std::vector<std::shared_ptr<AbstractCommand>> commands;
    };
}

#endif //ENGINE24_COMMAND_H
