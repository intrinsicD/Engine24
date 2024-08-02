//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_VERTEXARRAYOBJECT_H
#define ENGINE24_VERTEXARRAYOBJECT_H

#include <vector>
#include "Attribute.h"


namespace Bcg {
    struct VertexArrayObject {
        unsigned int id = -1;

        operator bool() const {
            return id != -1;
        }

        VertexArrayObject();

        void create();

        void destroy();

        void bind();

        void unbind();

        void setAttribute(unsigned int index, unsigned int size, unsigned int type, bool normalized,
                          unsigned int stride, const void *pointer);

        void enableAttribute(unsigned int index);

        void disableAttribute(unsigned int index);

        std::vector<Attribute> attributes;
    };
}

#endif //ENGINE24_VERTEXARRAYOBJECT_H
