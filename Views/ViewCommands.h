//
// Created by alex on 31.07.24.
//

#ifndef ENGINE24_VIEWCOMMANDS_H
#define ENGINE24_VIEWCOMMANDS_H

#include "Command.h"

namespace Bcg {
    //Constructs a buffer if it does not already exist and binds it to the respective view
    struct SetupPointsView : public AbstractCommand {
        SetupPointsView(const std::string &property_name) : AbstractCommand("SetupPointsView"),
                                                            property_name(property_name) {

        }

        void execute() const override;

        std::string property_name;
    };

    struct SetupGraphView : public AbstractCommand {
        SetupGraphView(const std::string &property_name) : AbstractCommand("SetupGraphView"),
                                                            property_name(property_name) {

        }

        void execute() const override;

        std::string property_name;
    };
}

#endif //ENGINE24_VIEWCOMMANDS_H
