#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in VS_OUT {
    vec3 position;// camera‐space point center
    vec3 normal;// camera‐space normal
    vec3 color;// per‐point color
    float radius;// radius in camera space
} gs_in[];

out GS_OUT {
    vec3 fragNormal;// to fragment shader
    vec3 fragColor;// to fragment shader
    vec3 fragPosition;// to fragment shader (camera‐space)
    vec2 gsUV;// to fragment shader (for discarding fragments outside the quad)
} gs_out;

// Same uniform block as the vertex shader, so that we can get projection:
layout (std140) uniform Camera {
    mat4 view;// WORLD → CAMERA (not used directly here)
    mat4 projection;// CAMERA → CLIP
};

void main() {
    vec3 P = gs_in[0].position;// camera‐space point center
    vec3 N = normalize(gs_in[0].normal);
    vec3 C = gs_in[0].color;// per‐point color
    float R = gs_in[0].radius;// splat radius (camera space)

    // Build a camera‐facing basis (T, B) perpendicular to N:
    vec3 V = normalize(-P);// view direction from P to camera (0,0,0)
    vec3 T = normalize(cross(N, V));// tangent
    vec3 B = cross(T, N);// bitangent

    // Offsets of the four quad corners:
    vec2 offsets[4] = vec2[4](
    vec2(-R, -R),
    vec2(+R, -R),
    vec2(-R, +R),
    vec2(+R, +R)
    );

    // Emit the quad as a triangle strip of 4 verts:
    for (int i = 0; i < 4; ++i) {
        vec3 offsetWorld = T * offsets[i].x + B * offsets[i].y;
        vec3 cornerPos = P + offsetWorld;// camera‐space corner
        vec4 clipPos   = projection * vec4(cornerPos, 1.0);

        gl_Position       = clipPos;
        gs_out.fragPosition = cornerPos;// camera‐space POS
        gs_out.fragNormal   = N;// same normal everywhere
        gs_out.fragColor    = C;// same color everywhere
        gs_out.gsUV         = offsets[i] / R * 2.0 - 1.0;// from [0,1] to [-1, 1] range for fragment shader
        EmitVertex();
    }
    EndPrimitive();
}
