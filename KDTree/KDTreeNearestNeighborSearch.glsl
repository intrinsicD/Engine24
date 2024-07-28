#version 430 core

layout(local_size_x = 256) in;

struct Vector3 {
    float x, y, z;
};

// Functions to access and modify components
float get(Vector3 vec, int index) {
    if (index == 0) return vec.x;
    else if (index == 1) return vec.y;
    else return vec.z;
}

void set(inout Vector3 vec, int index, float value) {
    if (index == 0) vec.x = value;
    else if (index == 1) vec.y = value;
    else vec.z = value;
}

struct Node {
    Vector3 boundingBoxMin;
    Vector3 boundingBoxMax;
    int leftChild;
    int rightChild;
    int parent;
};

layout(std430, binding = 0) readonly buffer Points { Vector3 points[]; };
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };
layout(std430, binding = 2) writeonly buffer Result { int nearestPointIndices[]; };

uniform float maxDist;

shared float minDist[256];
shared int bestPointIndex[256];

float distanceSquared(Vector3 a, Vector3 b) {
    return (a.x - b.x) * (a.x - b.x) +
    (a.y - b.y) * (a.y - b.y) +
    (a.z - b.z) * (a.z - b.z);
}

void nearestNeighborSearch(int nodeIndex, Vector3 queryPoint, inout float minDist, inout int nearestPointIndex) {
    if (nodeIndex == -1) return;

    Node node = nodes[nodeIndex];
    float dist = distanceSquared(points[nodeIndex], queryPoint);
    if (dist < minDist) {
        minDist = dist;
        nearestPointIndex = nodeIndex;
    }

    int axis = nodeIndex % 3;
    float ax = get(queryPoint, axis);
    int nextNode = (ax < points[nodeIndex][axis]) ? node.leftChild : node.rightChild;
    int otherNode = (ax < points[nodeIndex][axis]) ? node.rightChild : node.leftChild;

    nearestNeighborSearch(nextNode, queryPoint, minDist, nearestPointIndex);

    if ((ax - points[nodeIndex][axis]) * (ax - points[nodeIndex][axis]) < minDist) {
        nearestNeighborSearch(otherNode, queryPoint, minDist, nearestPointIndex);
    }
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    Vector3 queryPoint = points[id];
    minDist[id] = maxDist * maxDist;
    bestPointIndex[id] = -1;

    nearestNeighborSearch(0, queryPoint, minDist[id], bestPointIndex[id]);

    nearestPointIndices[id] = bestPointIndex[id];
}