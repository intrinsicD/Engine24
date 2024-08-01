#version 330 core

in vec3 f_normal;
in vec3 f_color;
in vec3 f_world;

uniform vec3 lightPosition;

out vec4 FragColor;

void main()
{
    vec3 normal = normalize(f_normal);
    float diff = max(dot(normal, normalize(lightPosition - f_world)), 0);
    vec3 finalColor = diff * f_color;
    FragColor = vec4(finalColor, 1.0f);
}