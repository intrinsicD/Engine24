#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 vector;
layout (location = 3) in float lengths;

out Vertex{
    vec3 vector;
    vec3 color;
    float lengths;
} vertex;

uniform bool use_uniform_color = true;
uniform vec3 uniform_color = vec3(1.0, 1.0, 1.0);

uniform bool use_uniform_length = true;
uniform float uniform_length = 1.0;

void main(){
    gl_Position = vec4(position, 1.0f);

    vertex.vector = vector;
    if(use_uniform_color){
        vertex.color = uniform_color;
    }else{
        vertex.color = color;
    }

    if(use_uniform_length){
        vertex.lengths = uniform_length;
    }else{
        vertex.lengths = lengths;
    }
}

