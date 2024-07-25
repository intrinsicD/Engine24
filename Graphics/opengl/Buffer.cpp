//
// Created by alex on 16.07.24.
//

#include "Buffer.h"
#include "glad/gl.h"

namespace Bcg {
    void Buffer::create() {
        if (id == -1) {
            glGenBuffers(1, &id);
        }
    }

    void Buffer::destroy() {
        if (id != -1) {
            glDeleteBuffers(1, &id);
            id = -1;
        }
    }

    void Buffer::bind() {
        glBindBuffer(target, id);
    }

    void Buffer::bind_base(unsigned int index) {
        binding_point = index;
        glBindBufferBase(target, index, id);
    }

    void Buffer::unbind() {
        glBindBuffer(target, 0);
    }

    void Buffer::buffer_data(const void *data, unsigned int size_bytes, unsigned int usage_) {
        usage = static_cast<Usage>(usage_);
        glBufferData(target, size_bytes, data, usage);
    }

    void Buffer::buffer_sub_data(const void *data, unsigned int size_bytes, unsigned int offset) {
        glBufferSubData(target, offset, size_bytes, data);
    }

    void Buffer::get_buffer_sub_data(void *data, unsigned int size_bytes, unsigned int offset) {
        glGetBufferSubData(target, offset, size_bytes, data);
    }

    ArrayBuffer::ArrayBuffer() {
        target = Target::ARRAY_BUFFER;
    }

    ElementArrayBuffer::ElementArrayBuffer() {
        target = Target::ELEMENT_ARRAY_BUFFER;
    }

    ShaderStorageBuffer::ShaderStorageBuffer() {
        target = Target::SHADER_STORAGE_BUFFER;
    }

    UniformBuffer::UniformBuffer() {
        target = Target::UNIFORM_BUFFER;
    }
}