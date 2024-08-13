//
// Created by alex on 30.07.24.
//

#include "PointCloudCompute.h"
#include "Buffer.h"
#include "Program.h"
#include "Logger.h"
#include "PropertyEigenMap.h"
#include "OpenGLState.h"

namespace Bcg {
    Property<NormalType> ComputeVertexNormals(entt::entity entity_id, PropertyContainer &vertices, int k) {
        const char *computeShaderSource = R"(
            #version 430 core

            layout(local_size_x = 256) in;

            struct Vector3 {
                float x, y, z;
            };

            vec3 Get(Vector3 p){
                return vec3(p.x, p.y, p.z);
            }

            struct Node {
                Vector3 boundingBoxMin;
                Vector3 boundingBoxMax;
                uint leftChild;
                uint rightChild;
                uint parent;
            };


            layout(std430, binding = 0) readonly buffer Points { Vector3 points[]; };
            layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };

            const float maxDist = 3.402823466e+38;

            uniform uint num_closest_points; // Changed to constant size

            layout(std430, binding = 2) writeonly buffer Result { uint nearestPoints[]; };
            layout (std430, binding = 3) writeonly buffer VertexNormals { Vector3 normals[]; };

            shared float nearestDists[256][10]; // Assuming max 10 closest points
            shared uint nearestPointsIndices[256][10]; // Assuming max 10 closest points

            float distanceSquared(vec3 a, vec3 b) {
                vec3 diff = a - b;
                return dot(diff, diff);
            }

            void insertSorted(uint pointIndex, float dist, inout uint nearestPoints[], inout float nearestDists[], uint num_closest_points) {
                for (uint i = 0; i < num_closest_points; ++i) {
                    if (dist < nearestDists[i]) {
                        for (uint j = num_closest_points - 1; j > i; --j) {
                            nearestDists[j] = nearestDists[j - 1];
                            nearestPoints[j] = nearestPoints[j - 1];
                        }
                        nearestDists[i] = dist;
                        nearestPoints[i] = pointIndex;
                        break;
                    }
                }
            }

            void kNearestNeighbors(uint nodeIndex, vec3 queryPoint, uint num_closest_points, inout uint nearestPoints[], inout float nearestDists[10]) {
                if (nodeIndex == -1) return;

                Node node = nodes[nodeIndex];
                vec3 otherPoint = Get(points[nodeIndex]);
                float dist = distanceSquared(otherPoint, queryPoint);
                insertSorted(nodeIndex, dist, nearestPoints, nearestDists, num_closest_points);

                uint axis = nodeIndex % 3;
                float ax = queryPoint[axis];
                float ay = otherPoint[axis];
                uint nextNode = (ax < ay) ? node.leftChild : node.rightChild;
                uint otherNode = (ax > ay) ? node.rightChild : node.leftChild;

                kNearestNeighbors(nextNode, queryPoint, num_closest_points, nearestPoints, nearestDists);

                if ((ax - ay) * (ax - ay) < nearestDists[num_closest_points - 1]) {
                    kNearestNeighbors(otherNode, queryPoint, num_closest_points, nearestPoints, nearestDists);
                }
            }

            mat3 computeCovarianceMatrix(vec3 queryPoint, uint num_closest_points, uint nearestPoints[], Vector3 points[]) {
                mat3 covariance = mat3(0.0);
                for (uint i = 0; i < num_closest_points; i++) {
                    vec3 knnPoint = Get(points[nearestPoints[i]]);

                    vec3 centeredPoint = knnPoint - queryPoint;
                    covariance[0][0] += centeredPoint.x * centeredPoint.x;
                    covariance[0][1] += centeredPoint.x * centeredPoint.y;
                    covariance[0][2] += centeredPoint.x * centeredPoint.z;

                    covariance[1][0] += centeredPoint.y * centeredPoint.x;
                    covariance[1][1] += centeredPoint.y * centeredPoint.y;
                    covariance[1][2] += centeredPoint.y * centeredPoint.z;

                    covariance[2][0] += centeredPoint.z * centeredPoint.x;
                    covariance[2][1] += centeredPoint.z * centeredPoint.y;
                    covariance[2][2] += centeredPoint.z * centeredPoint.z;
                }

                if(num_closest_points > 2){
                    covariance /= (num_closest_points - 1);
                }
                return covariance;
            }

            vec3 computeEigenvalues(mat3 matrix) {
                float m11 = matrix[0][0];
                float m12 = matrix[0][1];
                float m13 = matrix[0][2];
                float m21 = matrix[1][0];
                float m22 = matrix[1][1];
                float m23 = matrix[1][2];
                float m31 = matrix[2][0];
                float m32 = matrix[2][1];
                float m33 = matrix[2][2];

                float p1 = m12 * m12 + m13 * m13 + m23 * m23;
                if (p1 == 0.0) {
                    float eig1 = m11;
                    float eig2 = m22;
                    float eig3 = m33;
                    if (eig1 <= eig2 && eig1 <= eig3)
                    return vec3(1.0, 0.0, 0.0);
                    else if (eig2 <= eig1 && eig2 <= eig3)
                    return vec3(0.0, 1.0, 0.0);
                    else
                    return vec3(0.0, 0.0, 1.0);
                }

                float q = (m11 + m22 + m33) / 3.0;
                float p2 = (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q) + (m33 - q) * (m33 - q) + 2.0 * p1;
                float p = sqrt(p2 / 6.0);

                mat3 B = (1.0 / p) * (matrix - q * mat3(1.0));

                float r = determinant(B) / 2.0;

                float phi;
                if (r <= -1.0)
                phi = 3.14159265358979323846 / 3.0;
                else if (r >= 1.0)
                phi = 0.0;
                else
                phi = acos(r) / 3.0;

                float eig1 = q + 2.0 * p * cos(phi);
                float eig3 = q + 2.0 * p * cos(phi + (2.0 * 3.14159265358979323846 / 3.0));
                float eig2 = 3.0 * q - eig1 - eig3;
                return vec3(eig1, eig2, eig3);
            }

            vec3 computeEigenvector(mat3 matrix, float eigenvalue){
                mat3 A = matrix - eigenvalue * mat3(1.0);
                return normalize(cross(A[0], A[1]));
            }

            void main() {
                uint v = gl_GlobalInvocationID.x;

                vec3 queryPoint = Get(points[v]);
                for (uint i = 0; i < num_closest_points; ++i) {
                    nearestDists[v][i] = maxDist * maxDist;
                    nearestPointsIndices[v][i] = -1;
                }

                kNearestNeighbors(0, queryPoint, num_closest_points, nearestPointsIndices[v], nearestDists[v]);

                mat3 covariance = computeCovarianceMatrix(queryPoint, num_closest_points, nearestPointsIndices[v], points);
                vec3 eigenvalues = computeEigenvalues(covariance);
                vec3 N = normalize(computeEigenvector(covariance, eigenvalues[2]));
                normals[v] = Vector3(N.x, N.y, N.z);
            }
        )";

        // Point coordinates
        auto points = vertices.get<PointType >("v:position");
        auto normals = vertices.get_or_add<NormalType >("v:normal");

        auto num_points = points.vector().size();

        OpenGLState openGlState(entity_id);
        auto b_positions = openGlState.get_buffer(points.name());
        if (!b_positions) {
            b_positions = ArrayBuffer();
            b_positions.create();
            b_positions.bind();
            b_positions.buffer_data(points.vector().data(),
                                    num_points * 3 * sizeof(float),
                                    Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(points.name(), b_positions);
        }

        struct Node {
            Vector<float, 3> boundingBoxMin;
            Vector<float, 3> boundingBoxMax;
            uint leftChild;
            uint rightChild;
            uint parent;
        };

        auto b_nodes = openGlState.get_buffer("KDTreeNodes");
        if (!b_nodes) {
            b_nodes = ShaderStorageBuffer();
            b_nodes.create();
            b_nodes.bind();
            b_nodes.buffer_data(nullptr,
                                num_points * sizeof(Node),
                                Buffer::Usage::DYNAMIC_DRAW);
            openGlState.register_buffer("KDTreeNodes", b_nodes);
        }

        auto b_knnresult = openGlState.get_buffer("v:knnresult");
        if (!b_knnresult) {
            b_knnresult = ShaderStorageBuffer();
            b_knnresult.create();
            b_knnresult.bind();
            openGlState.register_buffer("v:knnresult", b_positions);
        }

        b_knnresult.buffer_data(nullptr,
                                num_points * k * sizeof(unsigned int),
                                Buffer::Usage::STATIC_DRAW);

        auto b_normals = openGlState.get_buffer(normals.name());
        if (!b_normals) {
            b_normals = ArrayBuffer();
            b_normals.create();
            b_normals.bind();
            b_normals.buffer_data(normals.data(),
                                  normals.vector().size() * 3 * sizeof(float),
                                  Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(normals.name(), b_normals);
        }

        b_positions.target = Buffer::Target::SHADER_STORAGE_BUFFER;
        b_normals.target = Buffer::Target::SHADER_STORAGE_BUFFER;

        auto program = openGlState.get_compute_program("ComputePointCloudVertexNormals");
        if (!program) {
            if (!program.create_from_source(computeShaderSource)) {
                Log::Error("Failed to create compute shader program!\n");
                return normals;
            }
            openGlState.register_compute_program("ComputePointCloudVertexNormals", program);
        }

        program.use();

        // Bind buffers
        b_positions.bind_base(0);
        b_nodes.bind_base(1);
        b_knnresult.bind_base(2);
        b_normals.bind_base(3);

        program.set_uniform1ui("num_closest_points", k);

        program.dispatch(num_points, 1, 1);

        // Ensure all computations are done
        program.memory_barrier(ComputeShaderProgram::Barrier::SHADER_STORAGE_BARRIER_BIT);

        // Retrieve the computed normals
        b_normals.bind();
        b_normals.get_buffer_sub_data(normals.vector().data(), normals.vector().size() * 3 * sizeof(float));

        return normals;
    }
}