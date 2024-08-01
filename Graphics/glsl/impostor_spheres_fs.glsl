#version 330 core

in vec3 f_normal;
in vec3 f_color;
in vec4 f_view;
in vec4 f_world;
in float f_radius_view_space;

layout (std140) uniform Camera {
    mat4 view;
    mat4 projection;
};

uniform vec3 lightPosition;

out vec4 FragColor;

void main()
{
    vec2 coord = gl_PointCoord * 2.0 - 1.0;// Convert from [0,1] to [-1,1]
    float dist = dot(coord, coord);

    if (dist > 1.0) {
        discard;// Discard fragments outside the sphere
    }

    float z = sqrt(1.0 - dist);// Sphere depth
    float adjustedViewDepth = f_view.z + z * f_radius_view_space * 0.2; //f_view.z is negative, so add a value to mode towards the near plane
    vec4 adjustedClipSpacePos = projection * vec4(f_view.xy, adjustedViewDepth, f_view.w);
    float ndcDepth = adjustedClipSpacePos.z / adjustedClipSpacePos.w;
    gl_FragDepth = (ndcDepth * 0.5 + 0.5);

    vec3 normal = normalize(f_normal);
    float diff = max(dot(normal, normalize(lightPosition - f_world.xyz)), 0);
    vec3 finalColor = diff * f_color;

    if (length(normal) >= 0.5){
        FragColor = vec4(finalColor, 1.0f);
    } else {
        FragColor = vec4(f_color * z, 1.0f);
    }
}