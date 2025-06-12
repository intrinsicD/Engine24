//
// Created by alex on 24.07.24.
//

#include "Texture.h"
#include "glad/gl.h"

namespace Bcg {
    void Texture::create() {
        if (id == -1) {
            glGenTextures(1, &id);
        }
    }

    void Texture::destroy() {
        if (id != -1) {
            glDeleteTextures(1, &id);
            id = -1;
        }
    }

    void Texture::bind() const {
        if (id != -1) {
            glBindTexture(target, id);
        }
    }

    void Texture::unbind() const {
        glBindTexture(target, 0);
    }

    void Texture1D::set_data(const void *data, unsigned int width_) {
        width = width_;
        bind();
        glTexImage1D(target, level, internal_format, width, border, format, type, data);
        unbind();
    }

    void Texture2D::set_data(const void *data, unsigned int width_, unsigned int height_) {
        width = width_;
        height = height_;
        bind();
        glTexImage2D(target, level, internal_format, width, height, border, format, type, data);
        unbind();
    }

    void Texture3D::set_data(const void *data, unsigned int width_, unsigned int height_, unsigned int channels_) {
        width = width_;
        height = height_;
        channels = channels_;
        bind();
        glTexImage3D(target, level, internal_format, width, height, channels, border, format, type, data);
        unbind();
    }

    void Texture::use(unsigned int program_id) const {
        if (id != -1) {
            int id = glGetUniformLocation(program_id, shader_name.c_str());
            if (id >= 0) glUniform1i(id, unit);

            glActiveTexture(GL_TEXTURE0 + unit);
            bind();
        }
    }
}