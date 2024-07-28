#version 430 core

layout(local_size_x = 256) in;

struct Vector3 {
    float x, y, z;
};

struct Node {
    Vector3 boundingBoxMin;
    Vector3 boundingBoxMax;
    int leftChild;
    int rightChild;
    int parent;
};

layout(std430, binding = 0) readonly buffer Points { Vector3 points[]; };
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };

uniform Vector3 minBounds;
uniform Vector3 maxBounds;
layout(std430, binding = 2) writeonly buffer Result { int foundPoints[]; };

shared int foundCount;

bool isWithinBounds(Vector3 point, Vector3 minBounds, Vector3 maxBounds) {
    return point.x >= minBounds.x && point.x <= maxBounds.x &&
    point.y >= minBounds.y && point.y <= maxBounds.y &&
    point.z >= minBounds.z && point.z <= maxBounds.z;
}

void rangeSearch(int nodeIndex, Vector3 minBounds, Vector3 maxBounds, inout int foundPoints[], inout int foundCount) {
    if (nodeIndex == -1) return;

    Node node = nodes[nodeIndex];
    Vector3 point = points[nodeIndex];

    if (isWithinBounds(point, minBounds, maxBounds)) {
        int index = atomicAdd(foundCount, 1);
        foundPoints[index] = nodeIndex;
    }

    if (node.leftChild != -1) {
        rangeSearch(node.leftChild, minBounds, maxBounds, foundPoints, foundCount);
    }

    if (node.rightChild != -1) {
        rangeSearch(node.rightChild, minBounds, maxBounds, foundPoints, foundCount);
    }
}

void main() {
    uint id = gl_GlobalInvocationID.x;

    if (id == 0) {
        foundCount = 0;
    }
    barrier();

    Vector3 queryPoint = points[id];
    rangeSearch(0, minBounds, maxBounds, foundPoints, foundCount);

    barrier();

    // Copy found points to the result buffer in a single thread to avoid race conditions
    if (id == 0) {
        for (int i = 0; i < foundCount; i++) {
            foundPoints[i] = foundPoints[i];
        }
    }
}
