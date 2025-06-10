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

//Frisvad 2012 – “Building an Orthonormal Basis from a 3D Unit Vector Without Normalization” (JCGT)
void BuildOrthonormalBasis(in vec3 N, out vec3 T, out vec3 B) {
    if (N.z < -0.9999999) {
        T = vec3(0.0, -1.0, 0.0);
        B = vec3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + N.z);
        float b = -N.x * N.y * a;
        T = vec3(1.0 - N.x * N.x * a, b, -N.x);
        B = vec3(b, 1.0 - N.y * N.y * a, -N.y);
    }
}

void main() {
    vec3 P = gs_in[0].position;// camera‐space point center
    vec3 N = gs_in[0].normal;// camera‐space normal (assumed to be normalized)
    vec3 C = gs_in[0].color;// per‐point color
    float R = gs_in[0].radius;// splat radius (camera space)

    // Build a camera‐facing basis (T, B) perpendicular to N:
    vec3 V = normalize(-P);// view direction from P to camera (0,0,0)

    vec3 T;
    vec3 B;
    BuildOrthonormalBasis(N, T, B);// ensure T and B are orthonormal

    // Offsets of the four quad corners (centered on P):
    vec2 offsets[4] = vec2[4](
        vec2(-0.5 * R, -0.5 * R),
        vec2(+0.5 * R, -0.5 * R),
        vec2(-0.5 * R, +0.5 * R),
        vec2(+0.5 * R, +0.5 * R)
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
        gs_out.gsUV         = offsets[i] / (0.5 * R); // <-- fix: normalize to [-1, 1] for circular splat
        EmitVertex();
    }
    EndPrimitive();
}