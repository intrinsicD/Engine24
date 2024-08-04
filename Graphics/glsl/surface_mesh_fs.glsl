#version 330 core

in vec3 f_normal;
in vec3 f_color;
in vec3 f_world;

uniform vec3 light_position;
uniform float min_color = 0.0f;
uniform float max_color = 1.0f;

out vec4 FragColor;

void main()
{
    vec3 normal = normalize(f_normal);
    float diff = max(dot(normal, normalize(light_position - f_world)), 0);
    vec3 finalColor = (f_color - min_color) / (max_color - min_color);
    finalColor = diff * finalColor;
    FragColor = vec4(finalColor, 1.0f);
}