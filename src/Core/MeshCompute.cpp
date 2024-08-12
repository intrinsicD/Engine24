//
// Created by alex on 21.06.24.
//

#include "MeshCompute.h"
#include "Buffer.h"
#include "Program.h"
#include "SurfaceMeshTriangles.h"
#include "Logger.h"
#include "PropertyEigenMap.h"
#include "OpenGLState.h"
#include "Engine.h"

namespace Bcg {
    VertexProperty<Vector<float, 3>> ComputeSurfaceMeshVertexNormals(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) return VertexProperty<Vector<float, 3>>();
        if (!Engine::has<SurfaceMesh>(entity_id)) return VertexProperty<Vector<float, 3>>();

        auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
        auto &vpoint = mesh.vpoint_.vector();
        auto &vconn = mesh.vconn_.vector();
        auto &hconn = mesh.hconn_.vector();
        auto &fconn = mesh.fconn_.vector();
        auto normals = mesh.vertex_property<Vector<float, 3>>("v:normal");

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

        auto b_vconns = openGlState.get_buffer(mesh.vconn_.name());
        if (!b_vconns) {
            b_vconns = ShaderStorageBuffer();
            b_vconns.create();
            b_vconns.bind();
            b_vconns.buffer_data(mesh.vconn_.data(),
                                 mesh.vconn_.vector().size() * sizeof(unsigned int),
                                 Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(mesh.vconn_.name(), b_vconns);
        }

        auto b_hconns = openGlState.get_buffer(mesh.hconn_.name());
        if (!b_hconns) {
            b_hconns = ShaderStorageBuffer();
            b_hconns.create();
            b_hconns.bind();
            b_hconns.buffer_data(mesh.hconn_.data(),
                                 mesh.hconn_.vector().size() * 4 * sizeof(unsigned int),
                                 Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(mesh.hconn_.name(), b_hconns);
        }

        auto b_fconns = openGlState.get_buffer(mesh.fconn_.name());
        if (!b_fconns) {
            b_fconns = ShaderStorageBuffer();
            b_fconns.create();
            b_fconns.bind();
            b_fconns.buffer_data(mesh.fconn_.data(),
                                 mesh.fconn_.vector().size() * sizeof(unsigned int),
                                 Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(mesh.fconn_.name(), b_fconns);
        }

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

        auto program = openGlState.get_compute_program("ComputeHalfedgeMeshVertexNormals");
        if (!program) {
            if (!program.create_from_file("../Shaders/glsl/surface_mesh_vertex_normal_cs.glsl")) {
                Log::Error("Failed to create compute shader program!\n");
                return normals;
            }
            openGlState.register_compute_program("ComputeHalfedgeMeshVertexNormals", program);
        }

        program.use();

        // Bind buffers
        b_positions.bind_base(0);
        b_vconns.bind_base(1);
        b_hconns.bind_base(2);
        b_fconns.bind_base(3);
        b_normals.bind_base(4);

        program.dispatch(mesh.vertices_size(), 1, 1);

        // Ensure all computations are done
        program.memory_barrier(ComputeShaderProgram::Barrier::SHADER_STORAGE_BARRIER_BIT);

        // Retrieve the computed normals
        b_normals.bind();
        b_normals.get_buffer_sub_data(normals.vector().data(), normals.vector().size() * 3 * sizeof(float));

        return normals;
    }

    FaceProperty<Vector<float, 3>> ComputeFaceNormals(entt::entity entity_id, SurfaceMesh &mesh) {
        const char *computeShaderSource = R"(
        #version 430 core
        layout (local_size_x = 1) in;

        struct Vector3{
            float x, y, z;
        };

        layout (std430, binding = 0) readonly buffer VertexPositions { Vector3 positions[]; };
        layout (std430, binding = 1) readonly buffer Indices { uvec3 indices[]; };
        layout (std430, binding = 2) writeonly buffer FaceNormals { Vector3 normals[]; };

        vec3 Get(Vector3 p){
            return vec3(p.x, p.y, p.z);
        }

        void main() {
            uint index = gl_GlobalInvocationID.x;
            uvec3 vertexIndices = indices[index];
            vec3 v0 = Get(positions[vertexIndices.x]);
            vec3 v1 = Get(positions[vertexIndices.y]);
            vec3 v2 = Get(positions[vertexIndices.z]);
            vec3 N = normalize(cross(v1 - v0, v2 - v0))
            normals[index] = Vector3(N.x, N.y, N.z);
        }
    )";

        auto f_normals = mesh.add_face_property<Vector<float,
                3 >>("f:normal");

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

        auto f_triangles = SurfaceMeshTriangles(mesh);
        auto b_triangles = openGlState.get_buffer(f_triangles.name());
        if (!b_triangles) {
            b_triangles = ElementArrayBuffer();
            b_triangles.create();
            b_triangles.bind();
            b_triangles.buffer_data(f_triangles.data(),
                                    f_triangles.vector().size() * 3 * sizeof(unsigned int),
                                    Buffer::STATIC_DRAW);
            openGlState.register_buffer(f_triangles.name(), b_triangles);
        }

        auto b_normals = openGlState.get_buffer(f_normals.name());
        if (!b_normals) {
            b_normals = ArrayBuffer();
            b_normals.create();
            b_normals.bind();
            b_normals.buffer_data(f_normals.data(),
                                  f_normals.vector().size() * 3 * sizeof(float),
                                  Buffer::STATIC_DRAW);
            openGlState.register_buffer(f_normals.name(), b_normals);
        }

        b_positions.target = Buffer::Target::SHADER_STORAGE_BUFFER;
        b_normals.target = Buffer::Target::SHADER_STORAGE_BUFFER;

        auto program = openGlState.get_compute_program("ComputeHalfedgeMeshFaceNormals");
        if (!program) {
            if (!program.create_from_source(computeShaderSource)) {
                Log::Error("Failed to create compute shader program!\n");
                return f_normals;
            }
            openGlState.register_compute_program("ComputeHalfedgeMeshFaceNormals", program);
        }

        program.use();

        // Bind buffers
        b_positions.bind_base(0);
        b_triangles.bind_base(1);
        b_normals.bind_base(2);

        program.dispatch(f_triangles.vector().size(), 1, 1);

        // Ensure all computations are done
        program.memory_barrier(ComputeShaderProgram::Barrier::SHADER_STORAGE_BARRIER_BIT);

        // Retrieve the computed normals
        b_normals.bind();
        b_normals.get_buffer_sub_data(f_normals.vector().data(), f_normals.vector().size() * 3 * sizeof(float));

        return f_normals;
    }

    void PT1A(entt::entity source_id, entt::entity target_id, float sigma2, float c) {
        const char *computeShaderSource = R"(
            #version 430 core
            layout (local_size_x = 1) in;

            struct Vector3 {
                float x, y, z;
            };

            layout (std430, binding = 0) readonly buffer FixedPoints { Vector3 X[]; };
            layout (std430, binding = 1) readonly buffer MovingPoints { Vector3 Y[]; };
            layout (std430, binding = 2) writeonly buffer ResponsibilityDenominator { float a[]; };
            layout (std430, binding = 3) writeonly buffer PT1 { float PT1[]; };

            uniform float variance; // σ^2
            uniform float constant_c; // c
            uniform uint numMovingPoints; // M

            vec3 get(Vector3 p) {
                return vec3(p.x, p.y, p.z);
            }

            void main() {
                uint n = gl_GlobalInvocationID.x;
                uint N = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

                // Ensure we are within bounds
                if (n >= N) return;

                vec3 xn = get(X[n]);
                float affinitySum = 0.0;

                for (uint m = 0; m < numMovingPoints; ++m) {
                    vec3 ym = get(Y[m]);
                    float norm = distance(xn, ym);
                    norm = norm * norm; // Compute the squared norm
                    affinitySum += exp(-norm / (2.0 * variance));
                }

                float an = affinitySum + constant_c;
                a[n] = an;
                PT1[n] = affinitySum / an;
            }
        )";

        // Use the compute shader program
        auto openGlState = OpenGLState(source_id);
        auto program = openGlState.get_compute_program("PT1A");
        if (!program) {
            if (!program.create_from_source(computeShaderSource)) {
                Log::Error("Failed to create compute shader program!\n");
                return;
            }
            openGlState.register_compute_program("PT1A", program);
        }

        // Set up FixedPoints
        auto &target = Engine::State().get<SurfaceMesh>(target_id);
        auto b_fixed_points = openGlState.get_buffer(target.vpoint_.name());
        if (!b_fixed_points) {
            b_fixed_points = ArrayBuffer();
            b_fixed_points.create();
            b_fixed_points.bind();
            b_fixed_points.buffer_data(target.positions().data(),
                                       target.positions().size() * 3 * sizeof(float),
                                       Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(target.vpoint_.name(), b_fixed_points);
        }

        // Set up MovingPoints
        auto &source = Engine::State().get<SurfaceMesh>(source_id);
        auto b_moving_points = openGlState.get_buffer(source.vpoint_.name());
        if (!b_moving_points) {
            b_moving_points = ArrayBuffer();
            b_moving_points.create();
            b_moving_points.bind();
            b_moving_points.buffer_data(source.positions().data(),
                                        source.positions().size() * 3 * sizeof(float),
                                        Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(source.vpoint_.name(), b_moving_points);
        }

        // Set up ResponsibilityDenominator buffer
        auto responsibilityDenominator = target.vprops_.get_or_add<float>("v:ResponsibilityDenominator", 0);
        auto b_responsibility_denominator = openGlState.get_buffer(responsibilityDenominator.name());
        if (!b_responsibility_denominator) {
            b_responsibility_denominator = ArrayBuffer();
            b_responsibility_denominator.create();
            b_responsibility_denominator.bind();
            b_responsibility_denominator.buffer_data(responsibilityDenominator.data(),
                                                     responsibilityDenominator.vector().size() * sizeof(float),
                                                     Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(responsibilityDenominator.name(), b_responsibility_denominator);
        }

        // Set up PT1 buffer
        auto pt1 = target.vprops_.get_or_add<float>("v:PT1", 0);
        auto b_pt1 = openGlState.get_buffer(pt1.name());
        if (!b_pt1) {
            b_pt1 = ArrayBuffer();
            b_pt1.create();
            b_pt1.bind();
            b_pt1.buffer_data(pt1.data(),
                              pt1.vector().size() * sizeof(float),
                              Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(pt1.name(), b_pt1);
        }

        // Bind buffers to shader storage binding points
        b_fixed_points.bind_base(0);
        b_moving_points.bind_base(1);
        b_responsibility_denominator.bind_base(2);
        b_pt1.bind_base(3);

        program.use();

// Set uniform values
        program.set_uniform1f("variance", sigma2);
        program.set_uniform1f("constant_c", c);
        program.set_uniform1ui("numMovingPoints", source.positions().size());


// Dispatch the compute shader
        program.dispatch(target.positions().size(), 1, 1);


// Make sure the computation is done before accessing the results
        program.memory_barrier(ComputeShaderProgram::SHADER_STORAGE_BARRIER_BIT);


// Read back results from ResponsibilityDenominator and PT1 buffers if needed...
        b_responsibility_denominator.bind();
        b_responsibility_denominator.get_buffer_sub_data(responsibilityDenominator.vector().data(),
                                                         responsibilityDenominator.vector().size() * sizeof(float));

        b_pt1.bind();
        b_pt1.get_buffer_sub_data(pt1.vector().data(), pt1.vector().size() * sizeof(float));
    }

    void P1PX(entt::entity source_id, entt::entity target_id, float sigma2, float c) {
        const char *computeShaderSource = R"(
        #version 430 core
        layout (local_size_x = 1) in;

        struct Vector3 {
            float x, y, z;
        };

        layout (std430, binding = 0) readonly buffer FixedPoints { Vector3 X[]; };
        layout (std430, binding = 1) readonly buffer MovingPoints { Vector3 Y[]; };
        layout (std430, binding = 2) readonly buffer ResponsibilityDenominator { float a[]; };
        layout (std430, binding = 3) writeonly buffer P1 { float P1[]; };
        layout (std430, binding = 4) writeonly buffer PX { Vector3 PX[]; };

        uniform float variance; // σ^2
        uniform float constant_c; // c
        uniform uint numFixedPoints; // N

        vec3 get(Vector3 p) {
            return vec3(p.x, p.y, p.z);
        }

        void main() {
            uint m = gl_GlobalInvocationID.x;
            uint M = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

            // Ensure we are within bounds
            if (m >= M) return;

            vec3 ym = get(Y[m]);
            float rowSum = 0.0;
            vec3 rowSumX = vec3(0.0);

            for (uint n = 0; n < numFixedPoints; ++n) {
                vec3 xn = get(X[n]);
                float norm = distance(xn, ym);
                norm = norm * norm; // Compute the squared norm
                float responsibility = exp(-norm / (2.0 * variance));
                rowSum += responsibility;
                rowSumX += responsibility * xn;
            }

            P1[m] = rowSum;
            PX[m] = rowSumX;
        }
    )";

        // Use the compute shader program
        auto openGlState = OpenGLState(source_id);
        auto program = openGlState.get_compute_program("P1PX");
        if (!program) {
            if (!program.create_from_source(computeShaderSource)) {
                Log::Error("Failed to create compute shader program!\n");
                return;
            }
            openGlState.register_compute_program("P1PX", program);
        }

        program.use();

        // Set up FixedPoints
        auto &target = Engine::State().get<SurfaceMesh>(target_id);
        auto b_fixed_points = openGlState.get_buffer(target.vpoint_.name());
        if (!b_fixed_points) {
            b_fixed_points = ArrayBuffer();
            b_fixed_points.create();
            b_fixed_points.bind();
            b_fixed_points.buffer_data(target.positions().data(),
                                       target.positions().size() * 3 * sizeof(float),
                                       Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(target.vpoint_.name(), b_fixed_points);
        }

        // Set up MovingPoints
        auto &source = Engine::State().get<SurfaceMesh>(source_id);
        auto b_moving_points = openGlState.get_buffer(source.vpoint_.name());
        if (!b_moving_points) {
            b_moving_points = ArrayBuffer();
            b_moving_points.create();
            b_moving_points.bind();
            b_moving_points.buffer_data(source.positions().data(),
                                        source.positions().size() * 3 * sizeof(float),
                                        Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(source.vpoint_.name(), b_moving_points);
        }

        // Set up ResponsibilityDenominator buffer
        auto responsibilityDenominator = target.vprops_.get_or_add<float>("v:ResponsibilityDenominator", 0);
        auto b_responsibility_denominator = openGlState.get_buffer(responsibilityDenominator.name());
        if (!b_responsibility_denominator) {
            b_responsibility_denominator = ArrayBuffer();
            b_responsibility_denominator.create();
            b_responsibility_denominator.bind();
            b_responsibility_denominator.buffer_data(responsibilityDenominator.data(),
                                                     responsibilityDenominator.vector().size() * sizeof(float),
                                                     Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(responsibilityDenominator.name(), b_responsibility_denominator);
        }

        // Set up P1 buffer
        auto p1 = source.vprops_.get_or_add<float>("v:P1", 0);
        auto b_p1 = openGlState.get_buffer(p1.name());
        if (!b_p1) {
            b_p1 = ArrayBuffer();
            b_p1.create();
            b_p1.bind();
            b_p1.buffer_data(p1.data(),
                             p1.vector().size() * sizeof(float),
                             Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(p1.name(), b_p1);
        }

        // Set up PX buffer
        auto px = source.vprops_.get_or_add<Vector<float, 3>>("v:PX", Vector<float, 3>::Zero());
        auto b_px = openGlState.get_buffer(px.name());
        if (!b_px) {
            b_px = ArrayBuffer();
            b_px.create();
            b_px.bind();
            b_px.buffer_data(px.data(),
                             px.vector().size() * 3 * sizeof(float),
                             Buffer::Usage::STATIC_DRAW);
            openGlState.register_buffer(px.name(), b_px);
        }

        // Bind buffers to shader storage binding points
        b_fixed_points.bind_base(0);
        b_moving_points.bind_base(1);
        b_responsibility_denominator.bind_base(2);
        b_p1.bind_base(3);
        b_px.bind_base(4);

        // Set uniform values
        program.set_uniform1f("variance", sigma2);
        program.set_uniform1f("constant_c", c);
        program.set_uniform1ui("numFixedPoints", target.positions().size());

        // Dispatch the compute shader
        program.dispatch(source.positions().size(), 1, 1);

        // Make sure the computation is done before accessing the results
        program.memory_barrier(ComputeShaderProgram::SHADER_STORAGE_BARRIER_BIT);

        // Optionally, read back results from P1 and PX buffers if needed
        b_p1.bind();
        b_p1.get_buffer_sub_data(p1.vector().data(), p1.vector().size() * sizeof(float));

        b_px.bind();
        b_px.get_buffer_sub_data(px.vector().data(), px.vector().size() * 3 * sizeof(float));
    }
}