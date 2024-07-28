//
// Created by alex on 28.07.24.
//

#include "KDTreeCompute.h"
#include "Logger.h"
#include "OpenGLState.h"

namespace Bcg {
    void BuildKDTReeCompute(entt::entity entity_id, SurfaceMesh &mesh) {
        const char *computeShaderSource = R"(
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
        layout(std430, binding = 1) writeonly buffer Nodes { Node nodes[]; };
        layout(std430, binding = 2) writeonly buffer Indices { int indices[]; };

        uniform int numPoints;
        uniform int threshold;

        void parallelRadixSort(int axis, int start, int end);
        void insertionSort(int axis, int start, int end);
        void buildKdTree(int nodeIndex, int depth, int start, int end);
        Vector3 computeBoundingBoxMin(int start, int end);
        Vector3 computeBoundingBoxMax(int start, int end);

        void main() {
            uint id = gl_GlobalInvocationID.x;
            if (id < numPoints) {
                indices[id] = int(id);
            }
            if (id == 0) {
                buildKdTree(0, 0, 0, numPoints);
            }
        }

        void parallelRadixSort(int axis, int start, int end) {
            const int BITS = 10; // Number of bits to sort at a time
            const int BUCKETS = 1 << BITS; // Number of buckets
            int size = end - start;

            shared int bucketOffsets[BUCKETS]; // Shared memory for bucket offsets
            shared int localOffsets[BUCKETS]; // Shared memory for local bucket offsets
            shared int tempIndices[256]; // Shared memory for temporary indices

            for (int shift = 0; shift < 32; shift += BITS) {
                // Step 1: Histogram
                if (gl_LocalInvocationID.x < BUCKETS) {
                    bucketOffsets[gl_LocalInvocationID.x] = 0;
                }
                barrier();

                for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
                    int value = floatBitsToInt(axis == 0 ? points[indices[start + i]].x :
                    axis == 1 ? points[indices[start + i]].y :
                    points[indices[start + i]].z);
                    int bucket = (value >> shift) & (BUCKETS - 1);
                    atomicAdd(bucketOffsets[bucket], 1);
                }
                barrier();

                // Step 2: Exclusive Scan
                if (gl_LocalInvocationID.x == 0) {
                    int sum = 0;
                    for (int i = 0; i < BUCKETS; ++i) {
                        int count = bucketOffsets[i];
                        bucketOffsets[i] = sum;
                        sum += count;
                    }
                }
                barrier();

                // Step 3: Local Prefix Sum
                for (uint i = gl_LocalInvocationID.x; i < size; i += gl_WorkGroupSize.x) {
                    int value = floatBitsToInt(axis == 0 ? points[indices[start + i]].x :
                    axis == 1 ? points[indices[start + i]].y :
                    points[indices[start + i]].z);
                    int bucket = (value >> shift) & (BUCKETS - 1);
                    int localOffset = atomicAdd(localOffsets[bucket], 1);
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

        void insertionSort(int axis, int start, int end) {
            for (int i = start + 1; i < end; i++) {
                int j = i;
                Vector3 temp = points[indices[i]];
                int tempIndex = indices[i];

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

        void buildKdTree(int nodeIndex, int depth, int start, int end) {
            if (start >= end) return;
            int axis = depth % 3;
            int mid = (start + end) / 2;
            if (end - start > threshold) parallelRadixSort(axis, start, end);
            else insertionSort(axis, start, end);
            nodes[nodeIndex].leftChild = 2 * nodeIndex + 1;
            nodes[nodeIndex].rightChild = 2 * nodeIndex + 2;
            nodes[nodeIndex].boundingBoxMin = computeBoundingBoxMin(start, end);
            nodes[nodeIndex].boundingBoxMax = computeBoundingBoxMax(start, end);
            buildKdTree(2 * nodeIndex + 1, depth + 1, start, mid);
            buildKdTree(2 * nodeIndex + 2, depth + 1, mid + 1, end);
        }

        Vector3 computeBoundingBoxMin(int start, int end) {
            Vector3 minVal = points[start];
            for (int i = start + 1; i < end; ++i) {
                minVal.x = min(minVal.x, points[i].x);
                minVal.y = min(minVal.y, points[i].y);
                minVal.z = min(minVal.z, points[i].z);
            }
            return minVal;
        }

        Vector3 computeBoundingBoxMax(int start, int end) {
            Vector3 maxVal = points[start];
            for (int i = start + 1; i < end; ++i) {
                maxVal.x = max(maxVal.x, points[i].x);
                maxVal.y = max(maxVal.y, points[i].y);
                maxVal.z = max(maxVal.z, points[i].z);
            }
            return maxVal;
        }
        )";

        OpenGLState openGlState(entity_id);
        auto b_positions = openGlState.get_buffer(mesh.vpoint_.name());
        if (!b_positions) {
            b_positions = ArrayBuffer();
            b_positions.create();
            b_positions.bind();
            b_positions.buffer_data(mesh.positions().data(),
                                    mesh.positions().size() * 3 * sizeof(float),
                                    Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(mesh.vpoint_.name(), b_positions);
        }

        struct Node {
            Vector<float, 3> boundingBoxMin;
            Vector<float, 3> boundingBoxMax;
            int leftChild;
            int rightChild;
            int parent;
        };

        auto b_nodes = openGlState.get_buffer("KDTreeNodes");
        if (!b_nodes) {
            b_nodes = ShaderStorageBuffer();
            b_nodes.create();
            b_nodes.bind();
            b_nodes.buffer_data(nullptr,
                                mesh.positions().size() * sizeof(Node),
                                Buffer::Usage::DYNAMIC_DRAW);
            openGlState.register_buffer("KDTreeNodes", b_nodes);
        }

        auto b_indices = openGlState.get_buffer("KDTreeIndices");
        if (!b_indices) {
            b_indices = ShaderStorageBuffer();
            b_indices.create();
            b_indices.bind();
            b_indices.buffer_data(nullptr,
                                  mesh.positions().size() * sizeof(unsigned int),
                                  Buffer::Usage::DYNAMIC_DRAW);
            openGlState.register_buffer("KDTreeIndices", b_indices);
        }

        auto program = openGlState.get_compute_program("ComputeKDTree");
        if (!program) {
            if (!program.create_from_source(computeShaderSource)) {
                Log::Error("Failed to create compute shader program!\n");
                return;
            }
            openGlState.register_compute_program("ComputeKDTree", program);
        }

        program.use();

        b_positions.bind_base(0);
        b_nodes.bind_base(1);
        b_indices.bind_base(2);

        program.set_uniform1ui("numPoints", mesh.positions().size());
        program.set_uniform1ui("threshold", 256);

        program.dispatch((mesh.positions().size() + 255) / 256, 1, 1);

        program.memory_barrier(ComputeShaderProgram::Barrier::SHADER_STORAGE_BARRIER_BIT);
    }
}