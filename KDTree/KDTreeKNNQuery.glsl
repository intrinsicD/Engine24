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

uniform int k;
layout(std430, binding = 2) writeonly buffer Result { int nearestPoints[]; };

shared float nearestDists[256][10];
shared int nearestPointsIndices[256][10];

float distanceSquared(Vector3 a, Vector3 b) {
    return (a.x - b.x) * (a.x - b.x) +
    (a.y - b.y) * (a.y - b.y) +
    (a.z - b.z) * (a.z - b.z);
}

void insertSorted(int pointIndex, float dist, inout int nearestPoints[], inout float nearestDists[], int k) {
    for (int i = 0; i < k; ++i) {
        if (dist < nearestDists[i]) {
            for (int j = k - 1; j > i; --j) {
                nearestDists[j] = nearestDists[j - 1];
                nearestPoints[j] = nearestPoints[j - 1];
            }
            nearestDists[i] = dist;
            nearestPoints[i] = pointIndex;
            break;
        }
    }
}

void kNearestNeighbors(int nodeIndex, Vector3 queryPoint, int k, inout int nearestPoints[], inout float nearestDists[]) {
    if (nodeIndex == -1) return;

    Node node = nodes[nodeIndex];
    float dist = distanceSquared(points[nodeIndex], queryPoint);
    insertSorted(nodeIndex, dist, nearestPoints, nearestDists, k);

    int axis = nodeIndex % 3;
    float ax = get(queryPoint, axis);
    int nextNode = (ax < points[nodeIndex][axis]) ? node.leftChild : node.rightChild;
    int otherNode = (ax < points[nodeIndex][axis]) ? node.rightChild : node.leftChild;

    kNearestNeighbors(nextNode, queryPoint, k, nearestPoints, nearestDists);

    if ((ax - points[nodeIndex][axis]) * (ax - points[nodeIndex][axis]) < nearestDists[k - 1]) {
        kNearestNeighbors(otherNode, queryPoint, k, nearestPoints, nearestDists);
    }
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    Vector3 queryPoint = points[id];
    for (int i = 0; i < k; ++i) {
        nearestDists[id][i] = maxDist * maxDist;
        nearestPointsIndices[id][i] = -1;
    }

    kNearestNeighbors(0, queryPoint, k, nearestPointsIndices[id], nearestDists[id]);

    for (int i = 0; i < k; i++) {
        nearestPoints[id * k + i] = nearestPointsIndices[id][i];
    }
}