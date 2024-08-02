//
// Created by alex on 31.07.24.
//

#include "Attribute.h"
#include "glad/gl.h"

namespace Bcg {
    void Attribute::set(const void *pointer) {
        glVertexAttribPointer(id, size, type, normalized, stride, pointer);
    }

    void Attribute::set_default(const float *values) {
        if (size == 1) {
            glVertexAttrib1f(id, *values);
        } else if (size == 2) {
            glVertexAttrib2f(id, values[0], values[1]);
        } else if (size == 3) {
            glVertexAttrib3f(id, values[0], values[1], values[2]);
        }
    }

    bool Attribute::is_enabled() const {
        GLint isEnabled;
        glGetVertexAttribiv(id, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &isEnabled);
        return isEnabled;
    }

    void Attribute::enable() {
        glEnableVertexAttribArray(id);
    }

    void Attribute::disable() {
        glDisableVertexAttribArray(id);
    }
}