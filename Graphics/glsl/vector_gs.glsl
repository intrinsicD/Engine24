#version 330 core
layout(points) in;
layout(line_strip, max_vertices=2) out;

layout (std140) uniform Camera {
    mat4 view;
    mat4 projection;
};

uniform mat4 model;

in Vertex{
    vec3 vector;
    vec3 color;
    float lengths;
} vertex[];

out vec3 f_color;

void EmitVertexVector(int id, mat4 mvp){
    vec3 P = gl_in[id].gl_Position.xyz;
    vec3 V = vertex[id].vector * vertex[id].lengths;
    f_color = vertex[id].color;

    gl_Position = mvp * vec4(P, 1.0);
    EmitVertex();

    gl_Position =  mvp * vec4(P + V, 1.0);
    EmitVertex();

    EndPrimitive();
}

void main(){
    mat4 mvp = projection * view * model;
    for (int i = 0; i < gl_in.length(); i++){
        EmitVertexVector(i, mvp);
    }
}