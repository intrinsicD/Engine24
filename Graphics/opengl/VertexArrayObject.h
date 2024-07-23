//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_VERTEXARRAYOBJECT_H
#define ENGINE24_VERTEXARRAYOBJECT_H

namespace Bcg {
    struct VertexArrayObject {
        unsigned int id;

        enum AttributeType {
            FLOAT = 0x1406,
            UNSIGNED_INT = 0x1405,
            UNSIGNED_BYTE = 0x1401
        };

        VertexArrayObject();

        void create();

        void destroy();

        void bind();

        void unbind();

        void setAttribute(unsigned int index, unsigned int size, unsigned int type, bool normalized, unsigned int stride, const void *pointer);

        void enableAttribute(unsigned int index);

        void disableAttribute(unsigned int index);
    };
}

#endif //ENGINE24_VERTEXARRAYOBJECT_H
