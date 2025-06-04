#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in float radius;

out VS_OUT {
    vec3 position;   // camera‐space position
    vec3 normal;     // camera‐space normal
    vec3 color;      // pass‐through color
    float radius;    // splat radius, assumed already in camera space
} vs_out;

// Uniform block containing view + projection.
// (In your C++ code, bind this UBO to binding point 0, for instance.)
layout (std140) uniform Camera {
    mat4 view;        // WORLD → CAMERA
    mat4 projection;  // CAMERA → CLIP
};

uniform mat4 model;                     // MODEL → WORLD
uniform bool use_uniform_radius = true;
uniform float uniform_radius = 1.0;

uniform bool use_uniform_color = true;
uniform vec3 uniform_color = vec3(1.0, 1.0, 1.0);

void main() {
    // 1) Transform position into camera space:
    mat4 model_view = view * model;
    vec4 view_pos = model_view * vec4(position, 1.0);

    vs_out.position = view_pos.xyz;

    // 2) Transform normal into camera space:
    //    (If uModel has non‐uniform scale, you’d want a proper normal‐matrix.
    //     For now we assume no skew/stretch or a uniform scale.)
    vs_out.normal = -mat3(transpose(inverse(model_view))) * normal;

    // 3) Decide color:
    if (use_uniform_color) {
        vs_out.color = uniform_color;
    } else {
        vs_out.color = color;
    }

    // 4) Decide radius (we assume “radius” and “uniform_radius” are already in camera space).
    if (use_uniform_radius) {
        vs_out.radius = uniform_radius;
    } else {
        vs_out.radius = radius;
    }

    // IMPORTANT: Do NOT write gl_Position here. The geometry shader
    // will expand each point into a splat‐quad and set gl_Position there.
}
