//
// Created by alex on 16.07.24.
//

#include "Buffer.h"
#include "glad/gl.h"

namespace Bcg {
    void Buffer::create() {
        glGenBuffers(1, &id);
    }

    void Buffer::destroy() {
        glDeleteBuffers(1, &id);
    }

    void Buffer::bind() {
        glBindBuffer(target, id);
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
        target = GL_ARRAY_BUFFER;
    }

    ElementArrayBuffer::ElementArrayBuffer() {
        target = GL_ELEMENT_ARRAY_BUFFER;
    }

    ShaderStorageBuffer::ShaderStorageBuffer() {
        target = GL_SHADER_STORAGE_BUFFER;
    }

    void ShaderStorageBuffer::bind_base(unsigned int index) {
        glBindBufferBase(target, index, id);
    }

    UniformBuffer::UniformBuffer() {
        target = GL_UNIFORM_BUFFER;
    }

    void UniformBuffer::bind_base(unsigned int index) {
        binding_point = index;
        glBindBufferBase(target, index, id);
    }
}