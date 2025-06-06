//
// Created by alex on 06.06.25.
//

#include "Colormap1D.h"
#include "glad/gl.h"

namespace Bcg {
    Colormap1D::Colormap1D(int resolution) : _resolution(resolution) {
        if (_resolution < 2) _resolution = 2;
        glGenTextures(1, &_texID);
        // Default to a “linear” ramp; subclasses can override getColor()
        generateTexture();
    }

    Colormap1D::~Colormap1D() {
        glDeleteTextures(1, &_texID);
    }

    /// Call this whenever you want to rebuild the 1D texture
    /// (e.g. after changing internal parameters in a subclass).
    void Colormap1D::generateTexture() {
        // 1) Sample getColor(t) at evenly‐spaced t in [0,1].
        std::vector<Vector<float, 4>> data;
        data.reserve(_resolution);
        for (int i = 0; i < _resolution; ++i) {
            float t = (float) i / float(_resolution - 1);
            data.push_back(getColor(t));
        }

        // 2) Bind and upload as GL_RGBA32F 1D texture.
        glBindTexture(GL_TEXTURE_1D, _texID);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

        // Allocate and fill. Each texel is 4 floats (RGBA).
        glTexImage1D(
                GL_TEXTURE_1D,
                0,                    // mipmap level
                GL_RGBA32F,           // internal format
                _resolution,          // width
                0,                    // border
                GL_RGBA,              // data format
                GL_FLOAT,             // data type
                data.data()           // pointer to CPU array
        );

        glBindTexture(GL_TEXTURE_1D, 0);
    }

    /// Bind the colormap to texture‐unit 'unit' (0..GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS−1).
    void Colormap1D::bind(int unit) const {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_1D, _texID);
    }

    /// Unbind from the given unit if you want:
    void Colormap1D::unbind(int unit) const {
        glActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_1D, 0);
    }

    Vector<float, 4> Colormap1D::getColorFromTable(float t, const Vector<float, 4> *table, size_t table_size) const{
        // Clamp t to [0,1]
        float x = std::clamp(t, 0.0f, 1.0f);

        // Map x to an index in [0, 255]:
        const size_t maxIndex = table_size - 1;
        const float scaled = x * maxIndex;
        size_t idx0 = static_cast<size_t>(std::floor(scaled));
        size_t idx1 = std::min(idx0 + 1, maxIndex);

        float frac = scaled - static_cast<float>(idx0);

        // Fetch both endpoints from the table and linearly interpolate
        const Vector<float, 4> &c0 = table[idx0];
        const Vector<float, 4> &c1 = table[idx1];
        return glm::mix(c0, c1, frac);
    }

    Vector<float, 4> JetColormap::getColor(float t) const {
        float r = std::clamp(1.0f - fabsf(4.0f * (t - 0.75f)), 0.0f, 1.0f);
        float g = std::clamp(1.0f - fabsf(4.0f * (t - 0.50f)), 0.0f, 1.0f);
        float b = std::clamp(1.0f - fabsf(4.0f * (t - 0.25f)), 0.0f, 1.0f);
        return Vector<float, 4>(r, g, b, 1.0f);
    }

    Vector<float, 4> ViridisColormap::getColor(float t) const {
        // A “Viridis”‐like piecewise ramp.
        float r = std::clamp(0.278f + 0.5f * t, 0.0f, 1.0f);
        float g = std::clamp(0.278f + 0.5f * t, 0.0f, 1.0f);
        float b = std::clamp(0.278f + 0.5f * t, 0.0f, 1.0f);
        return Vector<float, 4>(r, g, b, 1.0f);
    }

    Vector<float, 4> HotColdColormap::getColor(float t) const {
        if (t < 0.5f) {
            return Vector<float, 4>(0.0f, 0.0f, 1.0f - 2.0f * t, 1.0f); // Blue to Cyan
        } else {
            return Vector<float, 4>(2.0f * (t - 0.5f), 1.0f, 0.0f, 1.0f); // Cyan to Red
        }
    }

    Vector<float, 4> MagmaColormap::getColor(float t) const {
        // This is just a placeholder “approximate” ramp.
        // For a true Magma, you’d paste in the 256‐entry RGBA table or use
        // a known polynomial approximation. Here’s a rough piecewise approach:

        // The real Magma data is best taken from a 256‐entry LUT; for demo:
        float x = std::clamp(t, 0.0f, 1.0f);

        // Approximate by sampling a few control points:
        if (x < 0.25f) {
            // very dark purple → purple
            float a = x / 0.25f;
            return glm::mix(glm::vec4(0.001f,0.000f,0.014f,1.0f),
                            glm::vec4(0.362f,0.023f,0.355f,1.0f), a);
        }
        else if (x < 0.5f) {
            float a = (x - 0.25f)/0.25f;
            return glm::mix(glm::vec4(0.362f,0.023f,0.355f,1.0f),
                            glm::vec4(0.773f,0.114f,0.307f,1.0f), a);
        }
        else if (x < 0.75f) {
            float a = (x - 0.5f)/0.25f;
            return glm::mix(glm::vec4(0.773f,0.114f,0.307f,1.0f),
                            glm::vec4(0.997f,0.540f,0.176f,1.0f), a);
        }
        else {
            float a = (x - 0.75f)/0.25f;
            return glm::mix(glm::vec4(0.997f,0.540f,0.176f,1.0f),
                            glm::vec4(0.987f,0.991f,0.749f,1.0f), a);
        }
    }

    Vector<float, 4> InfernoColormap::getColor(float t) const {
        // Again, a rough 4‐segment approximation. For best results, use a 256 RGBA table.

        float x = std::clamp(t, 0.0f, 1.0f);
        if (x < 0.25f) {
            float a = x/0.25f;
            return glm::mix(glm::vec4(0.002f,0.001f,0.013f,1.0f),
                            glm::vec4(0.128f,0.043f,0.156f,1.0f), a);
        }
        else if (x < 0.5f) {
            float a = (x - 0.25f)/0.25f;
            return glm::mix(glm::vec4(0.128f,0.043f,0.156f,1.0f),
                            glm::vec4(0.628f,0.202f,0.345f,1.0f), a);
        }
        else if (x < 0.75f) {
            float a = (x - 0.5f)/0.25f;
            return glm::mix(glm::vec4(0.628f,0.202f,0.345f,1.0f),
                            glm::vec4(0.972f,0.394f,0.148f,1.0f), a);
        }
        else {
            float a = (x - 0.75f)/0.25f;
            return glm::mix(glm::vec4(0.972f,0.394f,0.148f,1.0f),
                            glm::vec4(0.988f,0.998f,0.644f,1.0f), a);
        }
    }

    Vector<float, 4> RdBuColormap::getColor(float t) const {
        // A simple interpolation between blue (t=0) → white (t=0.5) → red (t=1).
        float x = std::clamp(t, 0.0f, 1.0f);
        if (x < 0.5f) {
            float a = x / 0.5f;
            return glm::mix(glm::vec4(0.229f,0.299f,0.754f,1.0f),
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), a);
        } else {
            float a = (x - 0.5f) / 0.5f;
            return glm::mix(glm::vec4(1.0f,1.0f,1.0f,1.0f),
                            glm::vec4(0.706f,0.016f,0.150f,1.0f), a);
        }
    }

    /// 16‐entry Plasma samples (RGBA) taken from Matplotlib’s “plasma” at t = i/15.
    static constexpr std::array<glm::vec4, 16> PLASMA16_TABLE = {{
                                                                         { 0.050383f, 0.029803f, 0.527975f, 1.000000f },
                                                                         { 0.200445f, 0.017902f, 0.593364f, 1.000000f },
                                                                         { 0.312543f, 0.008239f, 0.635700f, 1.000000f },
                                                                         { 0.417642f, 0.000564f, 0.658390f, 1.000000f },
                                                                         { 0.517933f, 0.021563f, 0.654109f, 1.000000f },
                                                                         { 0.610667f, 0.090204f, 0.619951f, 1.000000f },
                                                                         { 0.692840f, 0.165141f, 0.564522f, 1.000000f },
                                                                         { 0.764193f, 0.240396f, 0.502126f, 1.000000f },
                                                                         { 0.826588f, 0.315714f, 0.441316f, 1.000000f },
                                                                         { 0.881443f, 0.392529f, 0.383229f, 1.000000f },
                                                                         { 0.928329f, 0.472975f, 0.326067f, 1.000000f },
                                                                         { 0.965024f, 0.559118f, 0.268513f, 1.000000f },
                                                                         { 0.988260f, 0.652325f, 0.211364f, 1.000000f },
                                                                         { 0.994141f, 0.753137f, 0.161404f, 1.000000f },
                                                                         { 0.977995f, 0.861432f, 0.142808f, 1.000000f },
                                                                         { 0.940015f, 0.975158f, 0.131326f, 1.000000f }
                                                                 }};


    Vector<float, 4> PlasmaColormap::getColor(float t) const {
        // This is a placeholder; you can replace it with actual Plasma colors.
        return Colormap1D::getColorFromTable(t, PLASMA16_TABLE.data(), PLASMA16_TABLE.size());
    }
}