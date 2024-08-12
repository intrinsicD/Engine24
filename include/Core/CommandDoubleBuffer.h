//
// Created by alex on 12.08.24.
//

#ifndef ENGINE24_COMMANDDOUBLEBUFFER_H
#define ENGINE24_COMMANDDOUBLEBUFFER_H

#include "CommandBuffer.h"

namespace Bcg{
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

#endif //ENGINE24_COMMANDDOUBLEBUFFER_H
