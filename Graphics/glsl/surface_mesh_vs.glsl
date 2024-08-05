#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

layout (std140) uniform Camera {
    mat4 view;
    mat4 projection;
};

uniform mat4 model;
uniform bool use_uniform_color = true;
uniform vec3 uniform_color = vec3(1.0, 1.0, 1.0);

out vec3 f_normal;
out vec3 f_world;
out vec3 f_color;

void main()
{
    if (use_uniform_color) {
        f_color = uniform_color;
    } else {
        f_color = aColor;
    }
    f_normal = mat3(transpose(inverse(model))) * aNormal;
    f_world = (model * vec4(aPos, 1.0)).xyz;
    gl_Position = projection * view * vec4(f_world, 1.0);
}