#version 430 core

layout(location = 0) in vec3 aPosition;

layout (std140) uniform Camera {
    mat4 view;        // WORLD → CAMERA
    mat4 projection;  // CAMERA → CLIP
};

uniform mat4 model;                     // MODEL → WORLD

void main() {
    gl_Position = projection * view * model * vec4(aPosition, 1.0);
}