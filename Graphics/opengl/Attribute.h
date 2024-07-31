//
// Created by alex on 31.07.24.
//

#ifndef ENGINE24_ATTRIBUTE_H
#define ENGINE24_ATTRIBUTE_H

namespace Bcg {
    struct Attribute {
        unsigned int id;
        unsigned int size;
        enum Type {
            FLOAT = 0x1406,
            UNSIGNED_INT = 0x1405,
            UNSIGNED_BYTE = 0x1401
        } type;
        bool normalized;
        unsigned int stride;
        const char *shader_name;
        const char *bound_buffer_name;

        void set(const void *pointer);

        void set_default(const float *values);

        bool is_enabled() const;

        void enable();

        void disable();
    };
}

#endif //ENGINE24_ATTRIBUTE_H
