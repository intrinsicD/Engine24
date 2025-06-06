//
// Created by alex on 06.06.25.
//

#ifndef ENGINE24_COLORMAP1D_H
#define ENGINE24_COLORMAP1D_H

#include "MatVec.h"

namespace Bcg {
    class Colormap1D {
    public:
        /// Construct with a given number of samples (texture width).
        /// Default = 256.  Must be ≥2.
        explicit Colormap1D(int resolution = 256);

        virtual ~Colormap1D();

        /// Returns the OpenGL texture ID (GL_TEXTURE_1D).
        unsigned int textureID() const { return _texID; }

        /// Call this whenever you want to rebuild the 1D texture
        /// (e.g. after changing internal parameters in a subclass).
        void generateTexture();

        /// Bind the colormap to texture‐unit 'unit' (0..GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS−1).
        void bind(int unit) const;

        /// Unbind from the given unit if you want:
        void unbind(int unit) const;

        /// Returns the nominal resolution (texture width).
        int resolution() const { return _resolution; }

        /// If you ever want to change 'resolution' at runtime,
        /// call setResolution(newRes) then generateTexture() again.
        void setResolution(int newRes) {
            if (newRes < 2) newRes = 2;
            if (newRes == resolution()) return;
            _resolution = newRes;
            generateTexture();
        }

    protected:
        /// Subclasses **must** override this.
        /// Input t ∈ [0,1]; return RGBA ∈ [0,1]^4.
        virtual Vector<float, 4> getColor(float t) const = 0;

        virtual Vector<float, 4> getColorFromTable(float t, const Vector<float, 4> *table, size_t table_size) const;
    private:
        unsigned int _texID = 0;
        int _resolution;
    };

    class JetColormap : public Colormap1D {
    public:
        explicit JetColormap(int resolution = 256) : Colormap1D(resolution) {}


    protected:
        /// “Jet”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };

    class ViridisColormap : public Colormap1D {
    public:
        explicit ViridisColormap(int resolution = 256) : Colormap1D(resolution) {}

    protected:
        /// “Viridis”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };

    class HotColdColormap : public Colormap1D {
    public:
        explicit HotColdColormap(int resolution = 256) : Colormap1D(resolution) {}

    protected:
        /// “HotCold”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };

    class MagmaColormap : public Colormap1D {
    public:
        explicit MagmaColormap(int resolution = 256) : Colormap1D(resolution) {}

    protected:
        /// “Magma”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };

    class InfernoColormap : public Colormap1D {
    public:
        explicit InfernoColormap(int resolution = 256) : Colormap1D(resolution) {}

    protected:
        /// “Inferno”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };

    class RdBuColormap : public Colormap1D {
    public:
        explicit RdBuColormap(int resolution = 256) : Colormap1D(resolution) {}

    protected:
        /// “RdBu”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };

    class PlasmaColormap : public Colormap1D {
    public:
        explicit PlasmaColormap(int resolution = 256) : Colormap1D(resolution) {}

    protected:
        /// “Set1”‐like piecewise ramp.
        Vector<float, 4> getColor(float t) const override;
    };
}

#endif //ENGINE24_COLORMAP1D_H
