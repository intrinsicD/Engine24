//
// Created by alex on 22.07.24.
//

#include "FrameBuffer.h"
#include "Logger.h"
#include "glad/gl.h"

namespace Bcg {
    void FrameBuffer::create() {
        glGenFramebuffers(1, &id);
    }

    void FrameBuffer::destroy() {
        if (id != 0) {
            glDeleteFramebuffers(1, &id);
            id = 0;
        }
    }

    void FrameBuffer::bind() const {
        glBindFramebuffer(target, id);
    }

    void FrameBuffer::unbind() const {
        glBindFramebuffer(target, 0);
    }

    std::string getStatusDestription(unsigned int status) {
        switch (status) {
            case GL_FRAMEBUFFER_UNDEFINED : {
                return "GL_FRAMEBUFFER_UNDEFINED is returned if the specified framebuffer is the default read or draw framebuffer, but the default framebuffer does not exist.";
            }
            case GL_FRAMEBUFFER_COMPLETE : {
                return "GL_FRAMEBUFFER_COMPLETE is returned if the framebuffer is complete.";
            }
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT : {
                return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT is returned if any of the framebuffer attachment points are framebuffer incomplete.";
            }
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT : {
                return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT is returned if the framebuffer does not have at least one image attached to it.";
            }
            case GL_FRAMEBUFFER_UNSUPPORTED : {
                return "GL_FRAMEBUFFER_UNSUPPORTED is returned if the combination of internal formats of the attached images violates an implementation-dependent set of restrictions.";
            }
            case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE : {
                return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE is returned if the value of GL_RENDERBUFFER_SAMPLES is not the same for all attached renderbuffers; if the value of GL_TEXTURE_SAMPLES is the not same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_RENDERBUFFER_SAMPLES does not match the value of GL_TEXTURE_SAMPLES.";
            }
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER : {
                return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER is returned if the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAW_BUFFERi.";
            }
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER : {
                return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER is returned if the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER.";
            }
            case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS : {
                return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS is returned if any framebuffer attachment is layered, and any populated attachment is not layered, or if all populated color attachments are not from textures of the same target.";
            }
            default: {
                return "Unknown framebuffer status.";
            }
        }
    }

    bool FrameBuffer::check() const {
        bind();
        unsigned int status = glCheckFramebufferStatus(target);
        bool complete = (status == GL_FRAMEBUFFER_COMPLETE);
        if (!complete) {
            Log::Error("Framebuffer not complete: " + getStatusDestription(status));
        }
        unbind();
        return complete;
    }

    void FrameBuffer::blit(int srcX0, int srcY0, int srcX1, int srcY1, int dstX0, int dstY0, int dstX1, int dstY1,
                           unsigned int mask, unsigned int filter, const FrameBuffer &other) const {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, id);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, other.id);
        glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    }

    void FrameBuffer::read_pixels(int x, int y, int width, int height, unsigned int format, unsigned int type,
                                  void *data) const {
        bind();
        glReadPixels(x, y, width, height, format, type, data);
        unbind();
    }

    unsigned int FrameBuffer::get_max_color_attachments() const {
        GLint maxColorAttachments;
        glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxColorAttachments);
        return maxColorAttachments;
    }

    bool FrameBuffer::add_texture_2d(const Texture2D &texture2D) {
        if (attachments.size() >= get_max_color_attachments()) return false;
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0 + attachments.size(), texture2D.target, texture2D.id,
                               texture2D.level);
        attachments.push_back(texture2D);
        return true;
    }
}