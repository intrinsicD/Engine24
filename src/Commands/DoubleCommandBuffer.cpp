//
// Created by alex on 29.11.24.
//

#include "CommandDoubleBuffer.h"

namespace Bcg{
    CommandBuffer &DoubleCommandBuffer::current() {
        return *p_current;
    }

    CommandBuffer &DoubleCommandBuffer::next() {
        return *p_next;
    }

    void DoubleCommandBuffer::swap_buffers() {
        std::swap(p_current, p_next);
    }

    void DoubleCommandBuffer::handle() {
        p_current->execute();
        p_current->clear();
        swap_buffers();
    }
}