#version 430 core

layout (local_size_x = 1) in;

struct Vector3{
    float x, y, z;
};

layout (std430, binding = 0) readonly buffer VertexPositions { Vector3 positions[]; };
layout (std430, binding = 1) readonly buffer VertexConnectivity { uint vconn[]; };
layout (std430, binding = 2) readonly buffer HalfedgeConnectivity { uvec4 hconn[]; };
layout (std430, binding = 3) readonly buffer FaceConnectivity { uint fconn[]; };

layout (std430, binding = 4) writeonly buffer VertexNormals { Vector3 normals[]; };

uint face(uint h) {
    return hconn[h].x;
}

uint to_vertex(uint h) {
    return hconn[h].y;
}

uint next_halfedge(uint h) {
    return hconn[h].z;
}

uint prev_halfedge(uint h) {
    return hconn[h].w;
}

uint opposite_halfedge(uint h) {
    return ((h & 1) == 1 ? h - 1 : h + 1);
}

uint ccw_rotated_halfedge(uint h) {
    return opposite_halfedge(prev_halfedge(h));
}

uint cw_rotated_halfedge(uint h) {
    return next_halfedge(opposite_halfedge(h));
}

vec3 Get(Vector3 p){
    return vec3(p.x, p.y, p.z);
}

void main() {
    uint v = gl_GlobalInvocationID.x;
    uint h = vconn[v];
    uint start = h;

    vec3 normal = vec3(0.0);
    vec3 v0 = Get(positions[v]);

    do {
        uint nh = next_halfedge(h);
        vec3 v1 = Get(positions[to_vertex(h)]);
        vec3 v2 = Get(positions[to_vertex(nh)]);
        normal += normalize(cross(v1 - v0, v2 - v0));
        h = cw_rotated_halfedge(h);
    } while (h != start && h != uint(-1));

    vec3 N = normalize(normal);
    normals[v] = Vector3(N.x, N.y, N.z);
}