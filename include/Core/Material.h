//
// Created by alex on 6/10/25.
//

#ifndef MATERIAL_H
#define MATERIAL_H

#include <variant>
#include "MatVec.h"
#include "Texture.h"

namespace Bcg {
    enum class ShadingModel { Phong, GGX, Flat };

    template<typename T>
    struct Parameter {
        std::string         uniform_name;     // GLSL uniform / sampler name
        T                   value;            // fallback uniform value
        Texture2D*          texture   = nullptr; // nullptr = don't sample
        SamplerDescriptor   sampler;          // filter/wrap settings
        Vector<float,2>     uv_scale{1.0f,1.0f};
        Vector<float,2>     uv_offset{0.0f,0.0f};
    };
    struct PhongShadingParams {
        Parameter<Vector<float,3>> ambient   {"uAmbient",   {0.1f,0.1f,0.1f}};
        Parameter<Vector<float,3>> diffuse   {"uDiffuse",   {0.8f,0.8f,0.8f}};
        Parameter<Vector<float,3>> specular  {"uSpecular",  {1.0f,1.0f,1.0f}};
        Parameter<float>           shininess {"uShininess", 32.0f};
    };

    struct GGXShadingParams {
        Parameter<Vector<float,3>> albedo    {"uAlbedo",    {0.8f,0.8f,0.8f}};
        Parameter<float>           metallic  {"uMetallic",  0.0f};
        Parameter<float>           roughness {"uRoughness", 0.5f};
        Parameter<float>           ao        {"uAO",        1.0f};
        Parameter<Vector<float,3>> emissive  {"uEmissive",  {0.0f,0.0f,0.0f}};
        Parameter<Vector<float,3>> normal    {"uNormalMap", {0.0f,0.0f,1.0f}};
    };

    struct FlatShadingParams {
        Parameter<Vector<float,3>> color     {"uColor",     {1.0f,1.0f,1.0f}};
    };

    using MaterialShadingParams = std::variant<
        PhongShadingParams,
        GGXShadingParams,
        FlatShadingParams
    >;

    class Material {
    public:
        unsigned int material_id;
        unsigned int program_id;
        constexpr ShadingModel shading_model;
        MaterialShadingParams params;
    };
}

#endif //MATERIAL_H
