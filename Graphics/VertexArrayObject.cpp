//
// Created by alex on 17.07.24.
//

#include "VertexArrayObject.h"
#include "glad/gl.h"

namespace Bcg{
    VertexArrayObject::VertexArrayObject() {
        id = 0;
    }

    void VertexArrayObject::create() {
        glGenVertexArrays(1, &id);
    }

    void VertexArrayObject::destroy() {
        glDeleteVertexArrays(1, &id);
    }

    void VertexArrayObject::bind() {
        glBindVertexArray(id);
    }

    void VertexArrayObject::unbind() {
        glBindVertexArray(0);
    }

    void VertexArrayObject::setAttribute(unsigned int index, unsigned int size, unsigned int type, bool normalized, unsigned int stride, const void *pointer){
        glVertexAttribPointer(index, size, type, normalized, stride, pointer);
    }

    void VertexArrayObject::enableAttribute(unsigned int index){
        glEnableVertexAttribArray(index);
    }

    void VertexArrayObject::disableAttribute(unsigned int index){
        glDisableVertexAttribArray(index);
    }
}