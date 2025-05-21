//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_TRIANGLEUTILS_H
#define ENGINE24_TRIANGLEUTILS_H

#include "Triangle.h"

namespace Bcg{
    template<typename T, int N>
    Eigen::Vector<T, 3> ToBarycentricCoords(const Triangle<T, N> &triangle, const Eigen::Vector<T, N> &point) {
        const Eigen::Vector<T, N>& v0 = triangle.u; // Corresponds to alpha
        const Eigen::Vector<T, N>& v1 = triangle.v; // Corresponds to beta
        const Eigen::Vector<T, N>& v2 = triangle.w; // Corresponds to gamma

        Eigen::Vector<T, N> e0 = v1 - v0;
        Eigen::Vector<T, N> e1 = v2 - v0;
        Eigen::Vector<T, N> p_minus_v0 = point - v0;

        T d00 = e0.dot(e0);
        T d01 = e0.dot(e1);
        T d11 = e1.dot(e1);
        T d20 = p_minus_v0.dot(e0);
        T d21 = p_minus_v0.dot(e1);

        T denom = d00 * d11 - d01 * d01;

        // A very small denom indicates collinearity or near-collinearity.
        // Standard float division by zero will produce Inf/NaN as appropriate.
        // No special check for std::abs(denom) < epsilon is strictly needed unless
        // a different behavior for degenerate triangles is required.

        T beta = (d11 * d20 - d01 * d21) / denom;  // Barycentric coordinate for v1
        T gamma = (d00 * d21 - d01 * d20) / denom; // Barycentric coordinate for v2
        T alpha = static_cast<T>(1.0) - beta - gamma;    // Barycentric coordinate for v0

        return Eigen::Vector<T, 3>(alpha, beta, gamma);
    }

    template<typename T, int N>
    Eigen::Vector<T, N> FromBarycentricCoords(const Triangle<T, N> &triangle, const Eigen::Vector<T, 3> &bc) {
        return bc[0] * triangle.u + bc[1] * triangle.v + bc[2] * triangle.w;
    }

    template<typename T, int N>
    Eigen::Vector<T, N> ClosestPoint(const Triangle<T, N> &triangle, const Eigen::Vector<T, N> &point) {
        //Christer Ericson method in Real-Time Collision Detection

        const Eigen::Vector<T, N>& a = triangle.u;
        const Eigen::Vector<T, N>& b = triangle.v;
        const Eigen::Vector<T, N>& c = triangle.w;

        // Use a small epsilon for squared norm checks to handle coincident points robustly.
        // This epsilon can be tuned; multiplying machine epsilon by a small factor is a common heuristic.
        const T epsilon = std::numeric_limits<T>::epsilon() * static_cast<T>(100.0);

        Eigen::Vector<T, N> ab = b - a;
        Eigen::Vector<T, N> ac = c - a;
        Eigen::Vector<T, N> ap = point - a;

        T d1 = ab.dot(ap); // Projection of AP onto AB, scaled by |AB|
        T d2 = ac.dot(ap); // Projection of AP onto AC, scaled by |AC|

        // Closest point is vertex A
        if (d1 <= static_cast<T>(0.0) && d2 <= static_cast<T>(0.0)) {
            return a;
        }

        Eigen::Vector<T, N> bp = point - b;
        T d3 = ab.dot(bp); // Projection of BP onto AB (related to BA.BP), scaled by |AB|
        T d4 = ac.dot(bp); // Projection of BP onto AC (from B's perspective), scaled by |AC|

        // Closest point is vertex B
        if (d3 >= static_cast<T>(0.0) && d4 <= d3) {
            return b;
        }

        // Check if P projects onto edge AB
        // vc is related to the signed area of triangle ABP; negative if P is on the "C-side" of AB.
        // More accurately, vc/((va+vb+vc)*Area(ABC)) is barycentric coord of C for point P's projection.
        T vc = d1 * d4 - d3 * d2;
        if (vc <= static_cast<T>(0.0) && d1 >= static_cast<T>(0.0) && d3 <= static_cast<T>(0.0)) {
            T ab_sq_norm = ab.squaredNorm();
            if (ab_sq_norm < epsilon) { return a; } // A and B are (nearly) coincident
            T v_param = d1 / ab_sq_norm; // Parameter for projection: P_on_AB = A + v_param * AB
            // Conditions d1>=0 and d3<=0 ensure 0 <= v_param <= 1
            return a + v_param * ab;
        }

        Eigen::Vector<T, N> cp = point - c;
        T d5 = ab.dot(cp); // Projection of CP onto AB
        T d6 = ac.dot(cp); // Projection of CP onto AC (related to CA.CP)

        // Closest point is vertex C
        if (d6 >= static_cast<T>(0.0) && d5 <= d6) {
            return c;
        }

        // Check if P projects onto edge AC
        T vb = d5 * d2 - d1 * d6;
        if (vb <= static_cast<T>(0.0) && d2 >= static_cast<T>(0.0) && d6 <= static_cast<T>(0.0)) {
            T ac_sq_norm = ac.squaredNorm();
            if (ac_sq_norm < epsilon) { return a; } // A and C are (nearly) coincident
            T w_param = d2 / ac_sq_norm; // Parameter for projection: P_on_AC = A + w_param * AC
            // Conditions d2>=0 and d6<=0 ensure 0 <= w_param <= 1
            return a + w_param * ac;
        }

        // Check if P projects onto edge BC
        T va = d3 * d6 - d5 * d4;
        // (d4 - d3) is (point-b).dot(c-b)
        // (d5 - d6) is (point-c).dot(b-c) which is -(point-c).dot(c-b)
        if (va <= static_cast<T>(0.0) && (d4 - d3) >= static_cast<T>(0.0) && (d5 - d6) >= static_cast<T>(0.0)) {
            Eigen::Vector<T, N> bc_vec = c - b;
            T bc_sq_norm = bc_vec.squaredNorm();
            if (bc_sq_norm < epsilon) { return b; } // B and C are (nearly) coincident
            T u_param_num = d4 - d3; // This is bc_vec.dot(point - b)
            T u_param = u_param_num / bc_sq_norm; // Parameter for projection: P_on_BC = B + u_param * BC
            // Conditions on (d4-d3) and (d5-d6) ensure 0 <= u_param <= 1
            return b + u_param * bc_vec;
        }

        // P projects inside the triangle face
        T denom_bary = va + vb + vc;
        // If denom_bary is zero, triangle is degenerate and P is on the line of degeneracy.
        // This situation should ideally be caught by previous edge/vertex checks.
        // If it's reached, results might be Inf/NaN if not handled.
        // A robust fallback could be added if denom_bary is very small.
        // However, for a non-degenerate triangle and P projecting inside, va,vb,vc > 0, so denom_bary > 0.

        T v_final_param = vb / denom_bary; // Barycentric V (for B), used as parameter for vector AB
        T w_final_param = vc / denom_bary; // Barycentric W (for C), used as parameter for vector AC

        // Closest point is P_proj = A + v_bary * AB + w_bary * AC
        return a + v_final_param * ab + w_final_param * ac;
    }

    template<typename T, int N>
    T Distance(const Triangle<T, N> &triangle, const Eigen::Vector<T, N> &point) {
        return (point - ClosestPoint(triangle, point)).norm();
    }

    template<typename T>
    Eigen::Vector<T, 3> Normal(const Triangle<T, 3> &triangle) {
        return (triangle.v - triangle.u).cross(triangle.w - triangle.u).normalized();
    }
}

#endif //ENGINE24_TRIANGLEUTILS_H
