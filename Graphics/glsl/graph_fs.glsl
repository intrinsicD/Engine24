#version 430 core

layout(std430, binding = 2) buffer EdgeColorBuffer {
    vec4 colors[];   // length = number of edges
};
layout(std430, binding = 3) buffer EdgeScalarBuffer {
    float scalars[];   // length = number of edges
};

uniform sampler1D colormap;  // bound to texture unit 0
uniform float     min_color;   // e.g. 0.0
uniform float     max_color;   // e.g. 1.0
uniform bool use_scalarfield;    // e.g. 0.0

uniform bool use_uniform_color;
uniform vec3 uniform_color;

out vec4 FragColor;

void main() {
    // Use edge color directly
    vec4 edgeColor = colors[ gl_PrimitiveID ];
    FragColor = edgeColor;

    if(use_scalarfield){
        // 1) Look up color for this edge:
        float scalar = scalars[ gl_PrimitiveID ];

        // 2) Normalize to [0,1]:
        float t = (scalar - min_color) / (max_color - min_color);
        t = clamp(t, 0.0, 1.0);

        // 3) Sample colormap 1D texture:
        vec4 cmapColor = texture(colormap, t);

        // 4) Output:
        FragColor = cmapColor; // Apply alpha from edge color
        return;
    }
    if(use_uniform_color){
        // Output uniform color
        FragColor = vec4(uniform_color, 1.0);
        return;
    }
}