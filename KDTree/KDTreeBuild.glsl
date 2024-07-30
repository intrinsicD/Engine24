/**
#version 430 core

layout (local_size_x = 256) in;

struct Vector3 {
    float x, y, z;
};

struct Node {
    Vector3 boundingBoxMin;
    Vector3 boundingBoxMax;
    uint leftChild;
    uint rightChild;
    uint parent;
};

layout (std430, binding = 0) readonly buffer Points { Vector3 points[]; };
layout (std430, binding = 1) buffer Nodes { Node nodes[]; };
layout (std430, binding = 2) buffer Indices { uint indices[]; };

uniform uint numPoints;
uniform uint threshold;
uniform uint passStart;
uniform uint passEnd;

shared uint bucketOffsets[1024]; // Adjusted for max buckets
shared uint localOffsets[1024]; // Adjusted for max buckets
shared uint tempIndices[256]; // Shared memory for temporary indices

void parallelRadixSort(uint axis, uint start, uint end);
void insertionSort(uint axis, uint start, uint end);
Vector3 computeBoundingBoxMin(uint start, uint end);
Vector3 computeBoundingBoxMax(uint start, uint end);

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id < numPoints) {
        indices[id] = id;
    }

    if (id >= passStart && id < passEnd) {
        uint nodeIndex = id;
        uint depth = 0;
        uint start = 0;
        uint end = numPoints;

        while (start < end) {
            uint axis = depth % 3;
            uint mid = (start + end) / 2;
            if (end - start > threshold) {
                parallelRadixSort(axis, start, end);
            } else {
                insertionSort(axis, start, end);
            }

            nodes[nodeIndex].leftChild = 2 * nodeIndex + 1;
            nodes[nodeIndex].rightChild = 2 * nodeIndex + 2;
            nodes[nodeIndex].boundingBoxMin = computeBoundingBoxMin(start, end);
            nodes[nodeIndex].boundingBoxMax = computeBoundingBoxMax(start, end);

            nodeIndex = 2 * nodeIndex + 1; // Move to the left child
            depth++;
            end = mid; // Update the end to mid for left child

            if (end <= start) {
                // Move to the right child if the left child is done
                nodeIndex = (nodeIndex - 1) / 2; // Move back to parent
                nodeIndex = 2 * nodeIndex + 2; // Move to the right child
                end = numPoints; // Reset end for right child
                start = mid + 1; // Update start to mid+1 for right child
            }
        }
    }
}

void parallelRadixSort(uint axis, uint start, uint end) {
    const uint BITS = 10; // Number of bits to sort at a time
    const uint BUCKETS = 1 << BITS; // Number of buckets
    uint size = end - start;

    for (uint shift = 0; shift < 32; shift += BITS) {
        // Step 1: Histogram
        if (gl_LocalInvocationID.x < BUCKETS) {
            bucketOffsets[gl_LocalInvocationID.x] = 0;
        }
        barrier();

        for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
            uint value = floatBitsToInt(axis == 0 ? points[indices[start + i]].x :
                                       axis == 1 ? points[indices[start + i]].y :
                                       points[indices[start + i]].z);
            uint bucket = (value >> shift) & (BUCKETS - 1);
            atomicAdd(bucketOffsets[bucket], 1);
        }
        barrier();

        // Step 2: Exclusive Scan
        if (gl_LocalInvocationID.x == 0) {
            uint sum = 0;
            for (uint i = 0; i < BUCKETS; ++i) {
                uint count = bucketOffsets[i];
                bucketOffsets[i] = sum;
                sum += count;
            }
        }
        barrier();

        // Step 3: Local Prefix Sum
        for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
            uint value = floatBitsToInt(axis == 0 ? points[indices[start + i]].x :
                                       axis == 1 ? points[indices[start + i]].y :
                                       points[indices[start + i]].z);
            uint bucket = (value >> shift) & (BUCKETS - 1);
            uint localOffset = atomicAdd(localOffsets[bucket], 1);
            tempIndices[bucketOffsets[bucket] + localOffset] = indices[start + i];
        }
        barrier();

        // Step 4: Write Sorted Values Back to Indices
        for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
            indices[start + i] = tempIndices[i];
        }
        barrier();
    }
}

void insertionSort(uint axis, uint start, uint end) {
    for (uint i = start + 1; i < end; i++) {
        uint j = i;
        Vector3 temp = points[indices[i]];
        uint tempIndex = indices[i];

        while (j > start && (
        (axis == 0 && points[indices[j - 1]].x > temp.x) ||
        (axis == 1 && points[indices[j - 1]].y > temp.y) ||
        (axis == 2 && points[indices[j - 1]].z > temp.z)
        )) {
            indices[j] = indices[j - 1];
            j--;
        }

        indices[j] = tempIndex;
    }
}

Vector3 computeBoundingBoxMin(uint start, uint end) {
    Vector3 minVal = points[start];
    for (uint i = start + 1; i < end; ++i) {
        minVal.x = min(minVal.x, points[i].x);
        minVal.y = min(minVal.y, points[i].y);
        minVal.z = min(minVal.z, points[i].z);
    }
    return minVal;
}

Vector3 computeBoundingBoxMax(uint start, uint end) {
    Vector3 maxVal = points[start];
    for (uint i = start + 1; i < end; ++i) {
        maxVal.x = max(maxVal.x, points[i].x);
        maxVal.y = max(maxVal.y, points[i].y);
        maxVal.z = max(maxVal.z, points[i].z);
    }
    return maxVal;
}*/

#version 430 core

layout (local_size_x = 256) in;

struct Vector3 {
    float x, y, z;
};

struct Node {
    Vector3 boundingBoxMin;
    Vector3 boundingBoxMax;
    uint leftChild;
    uint rightChild;
    uint parent;
};

layout (std430, binding = 0) readonly buffer Points { Vector3 points[]; };
layout (std430, binding = 1) buffer Nodes { Node nodes[]; };
layout (std430, binding = 2) buffer Indices { uint indices[]; };

uniform uint numPoints;
uniform uint threshold;
uniform uint passStart;
uniform uint passEnd;

shared uint bucketOffsets[1024]; // Adjusted for max buckets
shared uint localOffsets[1024]; // Adjusted for max buckets
shared uint tempIndices[256]; // Shared memory for temporary indices

void parallelRadixSort(uint axis, uint start, uint end);
void insertionSort(uint axis, uint start, uint end);

vec3[2] computeBoundingBox(uint start, uint end);

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id < numPoints) {
        indices[id] = id;
    }

    if (id >= passStart && id < passEnd) {
        uint nodeIndex = id;
        uint depth = 0;
        uint start = 0;
        uint end = numPoints;

        while (start < end) {
            uint axis = depth % 3;
            uint mid = (start + end) / 2;
            if (end - start > threshold) {
                parallelRadixSort(axis, start, end);
            } else {
                insertionSort(axis, start, end);
            }

            nodes[nodeIndex].leftChild = 2 * nodeIndex + 1;
            nodes[nodeIndex].rightChild = 2 * nodeIndex + 2;
            vec3[2] bounding_box = computeBoundingBoxMin(start, end);
            nodes[nodeIndex].boundingBoxMin = Vector3(bounding_box[0].x, bounding_box[0].y, bounding_box[0].z);
            nodes[nodeIndex].boundingBoxMax = Vector3(bounding_box[1].x, bounding_box[1].y, bounding_box[1].z);

            nodeIndex = 2 * nodeIndex + 1; // Move to the left child
            depth++;
            end = mid; // Update the end to mid for left child

            if (end <= start) {
                // Move to the right child if the left child is done
                nodeIndex = (nodeIndex - 1) / 2; // Move back to parent
                nodeIndex = 2 * nodeIndex + 2; // Move to the right child
                end = numPoints; // Reset end for right child
                start = mid + 1; // Update start to mid+1 for right child
            }
        }
    }
}

void parallelRadixSort(uint axis, uint start, uint end) {
    const uint BITS = 10; // Number of bits to sort at a time
    const uint BUCKETS = 1 << BITS; // Number of buckets
    uint size = end - start;

    for (uint shift = 0; shift < 32; shift += BITS) {
        // Step 1: Histogram
        if (gl_LocalInvocationID.x < BUCKETS) {
            bucketOffsets[gl_LocalInvocationID.x] = 0;
        }
        barrier();

        for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
            uint value = floatBitsToInt(axis == 0 ? points[indices[start + i]].x :
                                        axis == 1 ? points[indices[start + i]].y :
                                        points[indices[start + i]].z);
            uint bucket = (value >> shift) & (BUCKETS - 1);
            atomicAdd(bucketOffsets[bucket], 1);
        }
        barrier();

        // Step 2: Exclusive Scan
        if (gl_LocalInvocationID.x == 0) {
            uint sum = 0;
            for (uint i = 0; i < BUCKETS; ++i) {
                uint count = bucketOffsets[i];
                bucketOffsets[i] = sum;
                sum += count;
            }
        }
        barrier();

        // Step 3: Local Prefix Sum
        if (gl_LocalInvocationID.x < BUCKETS) {
            localOffsets[gl_LocalInvocationID.x] = 0;
        }
        barrier();

        for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
            uint value = floatBitsToInt(axis == 0 ? points[indices[start + i]].x :
                                        axis == 1 ? points[indices[start + i]].y :
                                        points[indices[start + i]].z);
            uint bucket = (value >> shift) & (BUCKETS - 1);
            uint localOffset = atomicAdd(localOffsets[bucket], 1);
            tempIndices[bucketOffsets[bucket] + localOffset] = indices[start + i];
        }
        barrier();

        // Step 4: Write Sorted Values Back to Indices
        for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
            indices[start + i] = tempIndices[i];
        }
        barrier();
    }
}

void insertionSort(uint axis, uint start, uint end) {
    for (uint i = start + 1; i < end; i++) {
        uint j = i;
        Vector3 temp = points[indices[i]];
        uint tempIndex = indices[i];

        while (j > start && (
        (axis == 0 && points[indices[j - 1]].x > temp.x) ||
        (axis == 1 && points[indices[j - 1]].y > temp.y) ||
        (axis == 2 && points[indices[j - 1]].z > temp.z)
        )) {
            indices[j] = indices[j - 1];
            j--;
        }

        indices[j] = tempIndex;
    }
}

vec3[2] computeBoundingBox(uint start, uint end) {
    vec3 minVal = Get(points[indices[start]]);
    vec3 maxVal = Get(points[indices[start]]);
    for (uint i = start + 1; i < end; ++i) {
        Vector3 p = points[indices[i]];
        minVal.x = min(minVal.x, p.x);
        minVal.y = min(minVal.y, p.y);
        minVal.z = min(minVal.z, p.z);

        maxVal.x = max(maxVal.x, p.x);
        maxVal.y = max(maxVal.y, p.y);
        maxVal.z = max(maxVal.z, p.z);
    }
    return vec3[2](minVal, maxVal);
}