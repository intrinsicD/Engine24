//
// Created by alex on 22.07.24.
//

#include "FrameBuffer.h"
#include "glad/gl.h"

namespace Bcg {
    Framebuffer::Framebuffer(uint32_t width, uint32_t height)
            : m_width(width), m_height(height) {
        invalidate();
    }

    Framebuffer::~Framebuffer() {
        glDeleteFramebuffers(1, &m_renderer_id);
        glDeleteTextures(1, &m_color_attachment);
        glDeleteTextures(1, &m_depth_attachment);
    }

// Move constructor
    Framebuffer::Framebuffer(Framebuffer&& other) noexcept
            : m_renderer_id(other.m_renderer_id),
              m_color_attachment(other.m_color_attachment),
              m_depth_attachment(other.m_depth_attachment),
              m_width(other.m_width),
              m_height(other.m_height) {
        // Invalidate the other object to prevent double deletion
        other.m_renderer_id = 0;
        other.m_color_attachment = 0;
        other.m_depth_attachment = 0;
    }

// Move assignment
    Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept {
        if (this != &other) {
            // Delete own resources first
            glDeleteFramebuffers(1, &m_renderer_id);
            glDeleteTextures(1, &m_color_attachment);
            glDeleteTextures(1, &m_depth_attachment);

            // Pilfer other's resources
            m_renderer_id = other.m_renderer_id;
            m_color_attachment = other.m_color_attachment;
            m_depth_attachment = other.m_depth_attachment;
            m_width = other.m_width;
            m_height = other.m_height;

            // Invalidate other
            other.m_renderer_id = 0;
            other.m_color_attachment = 0;
            other.m_depth_attachment = 0;
        }
        return *this;
    }


    void Framebuffer::invalidate() {
        // If we are resizing, delete old objects first
        if (m_renderer_id) {
            glDeleteFramebuffers(1, &m_renderer_id);
            glDeleteTextures(1, &m_color_attachment);
            glDeleteTextures(1, &m_depth_attachment);
        }

        // --- Create the Framebuffer Object ---
        glGenFramebuffers(1, &m_renderer_id);
        glBindFramebuffer(GL_FRAMEBUFFER, m_renderer_id);

        // --- Create the Color Attachment (for Entity IDs) ---
        glGenTextures(1, &m_color_attachment);
        glBindTexture(GL_TEXTURE_2D, m_color_attachment);
        // Use an integer format for the texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, m_width, m_height, 0, GL_RED_INTEGER, GL_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Attach the color texture to the framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color_attachment, 0);

        // --- Create the Depth Attachment ---
        glGenTextures(1, &m_depth_attachment);
        glBindTexture(GL_TEXTURE_2D, m_depth_attachment);
        // Use a standard depth format
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, m_width, m_height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr);

        // Attach the depth texture to the framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_depth_attachment, 0);

        // --- Final Check and Unbind ---
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            // Log an error here in a real engine
            // std::cerr << "Framebuffer is not complete!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind to avoid accidental rendering to it
    }

    void Framebuffer::resize(uint32_t width, uint32_t height) {
        if (width == 0 || height == 0 || (width == m_width && height == m_height)) {
            return;
        }
        m_width = width;
        m_height = height;
        invalidate();
    }

    void Framebuffer::bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, m_renderer_id);
        glViewport(0, 0, m_width, m_height);
    }

    void Framebuffer::unbind() const {
        // Bind the default framebuffer (the screen)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    int Framebuffer::read_pixel(uint32_t x, uint32_t y) {
        bind(); // Ensure we are reading from this framebuffer

        int pixel_data;
        // Read a 1x1 block of pixels at the specified coordinate
        glReadPixels(x, y, 1, 1, GL_RED_INTEGER, GL_INT, &pixel_data);

        unbind();
        return pixel_data;
    }
}