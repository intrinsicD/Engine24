//
// Created by alex on 31.07.24.
//

#include "Attribute.h"
#include "glad/gl.h"

namespace Bcg {
    void Attribute::set(const void *pointer) {
        glVertexAttribPointer(id, size, type, normalized, stride, pointer);
    }

    void Attribute::set_default(const float *values){
        glVertexAttrib3f(id, values[0], values[1], values[2]);
    }

    bool Attribute::is_enabled() const {
        GLint isEnabled;
        glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &isEnabled);
        return isEnabled;
    }

    void Attribute::enable() {
        glEnableVertexAttribArray(id);
    }

    void Attribute::disable() {
        glDisableVertexAttribArray(id);
    }
}