//
// Created by alex on 24.07.24.
//

#include "Texture.h"
#include "glad/gl.h"

namespace Bcg {
    void Texture::create() {
        if(id == -1){
            glGenTextures(1, &id);
        }
    }

    void Texture::destroy() {
        if (id != -1) {
            glDeleteTextures(1, &id);
            id = -1;
        }
    }
}