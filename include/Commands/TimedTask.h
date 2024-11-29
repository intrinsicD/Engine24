//
// Created by alex on 29.11.24.
//

#ifndef ENGINE24_TIMEDTASK_H
#define ENGINE24_TIMEDTASK_H

#include "Command.h"
#include "Timer.h"

namespace Bcg::Commands {
    struct Timed : public AbstractCommand {
        template<class Command>
        Timed(Command &&command) :  Timed(std::make_shared<Command>(std::forward<Command>(command))) {

        }

        Timed(std::shared_ptr<Commands::AbstractCommand> command) : AbstractCommand(command->name),
                                                                    cmd(std::move(command)) {

        }

        ~Timed() override = default;

        void execute() const override {
            if (cmd) {
                Timer timer;
                timer.start();
                cmd->execute();
                timer.stop();
                Log::Info(name + " took " + std::to_string(timer.delta) + " seconds");
            }
        }

        std::shared_ptr<AbstractCommand> cmd = nullptr;
    };
}

#endif //ENGINE24_TIMEDTASK_H
