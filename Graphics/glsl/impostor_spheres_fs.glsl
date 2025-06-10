#version 330 core

in vec3 f_color;
in vec4 f_view;
in vec4 f_world;
in vec3 f_normal;
in float f_radius_view_space;

layout (std140) uniform Camera {
    mat4 view;
    mat4 projection;
};

uniform float min_color = 0.0f;
uniform float max_color = 1.0f;
uniform vec3 light_position;

out vec4 FragColor;

void main()
{
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = dot(coord, coord);

    if (dist > 1.0) {
        discard; // Discard fragments outside the unit circle
    }

    float z = sqrt(1.0 - dist);
    float adjustedViewDepth = f_view.z + z * f_radius_view_space * 0.2;
    vec4 adjustedClipSpacePos = projection * vec4(f_view.xy, adjustedViewDepth, f_view.w);

    float ndcDepth = adjustedClipSpacePos.z / adjustedClipSpacePos.w;
    gl_FragDepth = (ndcDepth * 0.5 + 0.5);

    vec3 normal = normalize(f_normal);
    vec3 light_dir = normalize(light_position - f_world.xyz);
    float diff = max(dot(normal, light_dir), 0.0);

    vec3 normal_sphere = normalize(vec3(coord, z));
    diff = z * abs(normal_sphere.z);

    vec3 finalColor = (f_color - min_color) / (max_color - min_color);
    finalColor = diff * finalColor;
    FragColor = vec4(finalColor, 1.0);
}