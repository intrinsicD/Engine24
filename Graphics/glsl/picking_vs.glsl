#version 330 core
layout (location = 0) in vec3 position;

uniform mat4 model;

layout (std140) uniform Camera {
    mat4 view;        // WORLD → CAMERA
    mat4 projection;  // CAMERA → CLIP
};

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}