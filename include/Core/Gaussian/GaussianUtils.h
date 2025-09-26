#pragma once

#include "MatVec.h"

namespace Bcg {
    // Decode sigma from **log-stddevs**
    template<typename T>
    CUDA_HOST_DEVICE
    inline Vector<T, 3> decode_sigma_logstd(const Vector<T, 3> &s_log) {
        return Vector<T, 3>(std::exp(s_log.x), std::exp(s_log.y), std::exp(s_log.z)); // σ = exp(s)
    }

    // Log-pdf for numerical stability.
    // mean, scale_logstd in WORLD units; quat maps LOCAL→WORLD (unit or near-unit).
    template<typename T>
    CUDA_HOST_DEVICE
    inline T GaussianLogPdf(const Vector<T, 3> &x,
                            const Vector<T, 3> &mean,
                            const Vector<T, 3> &scale_logstd,
                            const Quaternion<T> &quat_local_to_world) {
        // 1) decode σ
        const Vector<T, 3> sigma = decode_sigma_logstd(scale_logstd);
        // 2) world→local: y = R^T (x - μ)
        const Vector<T, 3> y = rotate_by_unit_quat_conjugate(quat_local_to_world, x - mean);
        // 3) Mahalanobis^2 in local (diagonal Σ)
        const Vector<T, 3> inv_sigma = inv(sigma);
        const T md2 = (y.x * inv_sigma.x) * (y.x * inv_sigma.x)
                      + (y.y * inv_sigma.y) * (y.y * inv_sigma.y)
                      + (y.z * inv_sigma.z) * (y.z * inv_sigma.z);
        // 4) normalization
        const T log_two_pi = T(2.0) * std::numbers::pi_v<T>;
        const T log_norm = -T(0.5) * T(3) * std::log(log_two_pi) - (
                               std::log(sigma.x) + std::log(sigma.y) + std::log(sigma.z));
        return log_norm - T(0.5) * md2;
    }

    // Pdf (wraps log-pdf)
    template<typename T>
    CUDA_HOST_DEVICE
    inline T GaussianPdf(const Vector<T, 3> &x,
                         const Vector<T, 3> &mean,
                         const Vector<T, 3> &scale_logstd,
                         const Quaternion<T> &quat_local_to_world) {
        return std::exp(GaussianLogPdf(x, mean, scale_logstd, quat_local_to_world));
    }

    // Build A = R * diag(lambda) * R^T without forming R explicitly: use columns r1,r2,r3 = R e1, R e2, R e3
    // where lambda = (1/σ)^2 (componentwise).
    template<typename T>
    CUDA_HOST_DEVICE
    inline Matrix<T, 3, 3> build_A_from_quat_lambda(const Quaternion<T> &q, const Vector<T, 3> &lambda) {
        const Vector<T, 3> e1 = Vector<T, 3>(T(1), T(0), T(0));
        const Vector<T, 3> e2 = Vector<T, 3>(T(0), T(1), T(0));
        const Vector<T, 3> e3 = Vector<T, 3>(T(0), T(0), T(1));
        const Vector<T, 3> r1 = rotate_by_unit_quat(q, e1); // columns of R
        const Vector<T, 3> r2 = rotate_by_unit_quat(q, e2);
        const Vector<T, 3> r3 = rotate_by_unit_quat(q, e3);
        // A = λx r1 r1^T + λy r2 r2^T + λz r3 r3^T
        Matrix<T, 3, 3> A = mul(outer(r1, r1), lambda.x);
        A = add(A, mul(outer(r2, r2), lambda.y));
        A = add(A, mul(outer(r3, r3), lambda.z));
        return A;
    }

    template<typename T>
    struct GaussianEval {
        T log_pdf; // log p(x)
        T pdf; // p(x)
        Vector<T, 3> grad; // ∇_x p(x)
        Matrix<T, 3, 3> hess; // ∇^2_x p(x)
        // Optional: Hessian of log-pdf = -A (constant in x)
        Matrix<T, 3, 3> hess_log; // ∇^2_x log p(x) = -A
    };

    /// Evaluate pdf, gradient, and Hessian WITHOUT building Σ. Assumes 'scale_logstd' are log-σ.
    /// quat maps LOCAL→WORLD.
    template<typename T>
    CUDA_HOST_DEVICE
    GaussianEval<T> GaussianPdfGradHess(const Vector<T, 3> &x,
                                        const Vector<T, 3> &mean,
                                        const Vector<T, 3> &scale_logstd,
                                        const Quaternion<T> &quat_local_to_world) {
        // 1) decode σ and derived terms
        const Vector<T, 3> sigma = decode_sigma_logstd(scale_logstd);
        const Vector<T, 3> inv_sigma = inv(sigma);
        const Vector<T, 3> lambda = hadamard(inv_sigma, inv_sigma); // 1/σ^2

        // 2) world→local: y = R^T (x - μ)
        const Vector<T, 3> dx = x - mean;
        const Vector<T, 3> y = rotate_by_unit_quat_conjugate(quat_local_to_world, dx);

        // 3) Mahalanobis^2
        const T md2 = dot(hadamard(lambda, y), y); // y^T diag(λ) y

        // 4) log-pdf and pdf
        const T log_norm = -T(0.5) * T(3) * std::log(T(2.0) * std::numbers::pi_v<T>)
                           - (std::log(sigma.x) + std::log(sigma.y) + std::log(sigma.z));
        const T log_p = log_norm - T(0.5) * md2;
        const T p = std::exp(log_p);

        // 5) gradient of log-pdf in local, then to world
        // ∇_y log p = -diag(λ) y
        const Vector<T, 3> g_log_local = Vector<T, 3>(-lambda.x * y.x, -lambda.y * y.y, -lambda.z * y.z);
        // ∇_x log p = R * ∇_y log p
        const Vector<T, 3> g_log_world = rotate_by_unit_quat(quat_local_to_world, g_log_local);
        // ∇_x p = p * ∇_x log p
        const Vector<T, 3> grad = p * g_log_world;

        // 6) Hessian
        // ∇^2_x log p = -A, where A = R diag(λ) R^T  (x-independent)
        const Matrix<T, 3, 3> A = build_A_from_quat_lambda(quat_local_to_world, lambda);
        Matrix<T, 3, 3> Hlog; // = -A
        for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) Hlog[c][r] = -A[c][r];

        // ∇^2_x p = p * ( (∇log p)(∇log p)^T + ∇^2 log p )
        const Matrix<T, 3, 3> outer_glog = outer(g_log_world, g_log_world);
        Matrix<T, 3, 3> H = add(outer_glog, Hlog);
        H = mul(H, p);

        return {log_p, p, grad, H, Hlog};
    }

    // Orthonormal tangent basis from normal n
    template<typename T>
    CUDA_HOST_DEVICE inline void tangent_basis(const Vector<T, 3> &n, Vector<T, 3> &t1, Vector<T, 3> &t2) {
        // robust: pick the smallest component to avoid degeneracy
        if (std::fabs(n.x) > std::fabs(n.z)) {
            t1 = normalize(Vector<T, 3>(-n.y, n.x, T(0)));
        } else {
            t1 = normalize(Vector<T, 3>(T(0), -n.z, n.y));
        }
        t2 = cross(n, t1);
    }

    // ------------------------ principal curvatures ------------------------
    template<typename T>
    struct PrincipalCurvatures {
        T k1, k2; // principal curvatures (k1 >= k2 by convention)
        Vector<T, 3> e1, e2; // principal directions in world (tangent unit vectors)
    };

    /// Principal curvatures/directions of the Gaussian-pdf level set at x.
    /// Inputs:
    ///   x, mu                : world-space point and Gaussian mean
    ///   log_sigma            : log standard deviations (componentwise)
    ///   quat_local_to_world  : unit quaternion mapping local->world
    /// Returns k1 >= k2 and their directions (world).
    template<typename T>
    CUDA_HOST_DEVICE
    PrincipalCurvatures<T> GaussianLevelSetPrincipalCurvatures(
        const Vector<T, 3> &x,
        const Vector<T, 3> &mu,
        const Vector<T, 3> &log_sigma,
        const Quaternion<T> &quat_local_to_world) {
        // Decode σ and build precision A = R diag(1/σ^2) R^T
        const Vector<T, 3> sigma = exp_vec(log_sigma);
        const Vector<T, 3> invsig = inv(sigma);
        const Vector<T, 3> lambda = Vector<T, 3>(invsig.x * invsig.x, invsig.y * invsig.y, invsig.z * invsig.z);
        const Matrix<T, 3, 3> A = build_A_from_quat_lambda(quat_local_to_world, lambda);

        // g_log = -A (x - mu)  (world)
        const Vector<T, 3> d = x - mu;
        const Vector<T, 3> g_log = (matvec(A, d)) * T(-1);
        const T gnorm = norm(g_log);

        // If gradient is ~0 (numerical edge), return zero curvature and arbitrary directions
        if (gnorm <= T(1e-20)) {
            PrincipalCurvatures<T> out;
            out.k1 = out.k2 = T(0);
            out.e1 = Vector<T, 3>(T(1), T(0), T(0));
            out.e2 = Vector<T, 3>(T(0), T(1), T(0));
            return out;
        }

        // Unit normal of the level set (same direction as grad f ~ g_log)
        const Vector<T, 3> n = g_log * (T(1) / gnorm);

        // P = I - n n^T
        const Matrix<T, 3, 3> I = eye<T>();
        const Matrix<T, 3, 3> nnT = outer(n, n);
        const Matrix<T, 3, 3> P = sub(I, nnT);

        // M = g_log g_log^T - A
        const Matrix<T, 3, 3> M = sub(outer(g_log, g_log), A);

        // Project to tangent: H_T = P M P
        const Matrix<T, 3, 3> PM = {
            {
                {
                    P[0][0] * M[0][0] + P[0][1] * M[1][0] + P[0][2] * M[2][0],
                    P[0][0] * M[0][1] + P[0][1] * M[1][1] + P[0][2] * M[2][1],
                    P[0][0] * M[0][2] + P[0][1] * M[1][2] + P[0][2] * M[2][2]
                },
                {
                    P[1][0] * M[0][0] + P[1][1] * M[1][0] + P[1][2] * M[2][0],
                    P[1][0] * M[0][1] + P[1][1] * M[1][1] + P[1][2] * M[2][1],
                    P[1][0] * M[0][2] + P[1][1] * M[1][2] + P[1][2] * M[2][2]
                },
                {
                    P[2][0] * M[0][0] + P[2][1] * M[1][0] + P[2][2] * M[2][0],
                    P[2][0] * M[0][1] + P[2][1] * M[1][1] + P[2][2] * M[2][1],
                    P[2][0] * M[0][2] + P[2][1] * M[1][2] + P[2][2] * M[2][2]
                }
            }
        };
        const Matrix<T, 3, 3> H_T = {
            {
                {
                    PM[0][0] * P[0][0] + PM[0][1] * P[1][0] + PM[0][2] * P[2][0],
                    PM[0][0] * P[0][1] + PM[0][1] * P[1][1] + PM[0][2] * P[2][1],
                    PM[0][0] * P[0][2] + PM[0][1] * P[1][2] + PM[0][2] * P[2][2]
                },
                {
                    PM[1][0] * P[0][0] + PM[1][1] * P[1][0] + PM[1][2] * P[2][0],
                    PM[1][0] * P[0][1] + PM[1][1] * P[1][1] + PM[1][2] * P[2][1],
                    PM[1][0] * P[0][2] + PM[1][1] * P[1][2] + PM[1][2] * P[2][2]
                },
                {
                    PM[2][0] * P[0][0] + PM[2][1] * P[1][0] + PM[2][2] * P[2][0],
                    PM[2][0] * P[0][1] + PM[2][1] * P[1][1] + PM[2][2] * P[2][1],
                    PM[2][0] * P[0][2] + PM[2][1] * P[1][2] + PM[2][2] * P[2][2]
                }
            }
        };

        // Tangent basis {t1,t2}
        Vector<T, 3> t1, t2;
        tangent_basis(n, t1, t2);

        // 2x2 shape operator S = -(1/||g_log||) * [t_i^T H_T t_j]
        auto quad = [&](const Vector<T, 3> &a, const Matrix<T, 3, 3> &B, const Vector<T, 3> &b)-> T {
            // a^T B b
            return a.x * (B[0][0] * b.x + B[0][1] * b.y + B[0][2] * b.z) +
                   a.y * (B[1][0] * b.x + B[1][1] * b.y + B[1][2] * b.z) +
                   a.z * (B[2][0] * b.x + B[2][1] * b.y + B[2][2] * b.z);
        };
        const T a = quad(t1, H_T, t1);
        const T b = quad(t1, H_T, t2);
        const T c = quad(t2, H_T, t2);
        const T inv_g = T(-1) / gnorm; // minus sign from shape operator definition

        // Eigenvalues of symmetric 2x2 [[a,b],[b,c]]
        const T tr = a + c;
        const T det4 = (a - c) * (a - c) + T(4) * b * b;
        const T root = std::sqrt((det4 >= T(0)) ? det4 : T(0));
        T emax = T(0.5) * (tr + root);
        T emin = T(0.5) * (tr - root);

        PrincipalCurvatures<T> out;
        out.k1 = emax * inv_g;
        out.k2 = emin * inv_g;

        // Principal directions in world: project eigenvectors onto (t1,t2)
        // For [[a,b],[b,c]], eigenvector for emax is (b, emax - a) (unless b≈0)
        Vector<T, 3> v1_2d, v2_2d;
        if (std::fabs(b) > T(1e-20)) {
            v1_2d = Vector<T, 3>(b, emax - a, T(0));
            v2_2d = Vector<T, 3>(b, emin - a, T(0));
        } else {
            // matrix already diagonal in this basis
            v1_2d = Vector<T, 3>(T(1), T(0), T(0));
            v2_2d = Vector<T, 3>(T(0), T(1), T(0));
        }
        // Lift to 3D world: e = v.x * t1 + v.y * t2
        Vector<T, 3> e1w = normalize(t1 * v1_2d.x + t2 * v1_2d.y);
        Vector<T, 3> e2w = normalize(t1 * v2_2d.x + t2 * v2_2d.y);
        out.e1 = e1w;
        out.e2 = e2w;
        return out;
    }

    // ------------------------ outputs ------------------------
    template<typename T>
    struct MixtureCurvature {
        T F; // mixture value at x
        Vector<T, 3> grad; // ∇F
        T k1, k2; // principal curvatures, k1 >= k2
        Vector<T, 3> e1, e2; // principal directions (world)
    };

    // ------------------------ core routine ------------------------
    template<typename T>
    CUDA_HOST_DEVICE
    MixtureCurvature<T> GaussianMixtureLevelSetCurvatures(
        const Vector<T, 3> &x,
        const Vector<T, 3> *means, // array of size N
        const Vector<T, 3> *log_sigmas, // array of size N (log σ)
        const Quaternion<T> *quats, // array of size N (unit; local→world)
        const T *weights, // array of size N (can be nullptr => all 1)
        int N) {
        // Accumulate F, gradF, HessF
        T F = T(0);
        Vector<T, 3> g = Vector<T, 3>(T(0), T(0), T(0));
        Matrix<T, 3, 3> H{};
        H[0][0] = H[0][1] = H[0][2] = H[1][0] = H[1][1] = H[1][2] = H[2][0] = H[2][1] = H[2][2] = T(0);

        for (int i = 0; i < N; ++i) {
            const T wi = weights ? weights[i] : T(1);
            const Vector<T, 3> mu = means[i];
            const Vector<T, 3> sig = exp_vec(log_sigmas[i]); // σ_i
            const Vector<T, 3> invs = inv(sig);
            const Vector<T, 3> lam = Vector<T, 3>(invs.x * invs.x, invs.y * invs.y, invs.z * invs.z); // 1/σ^2
            const Quaternion<T> q = quats[i];

            // displacement and local coords
            const Vector<T, 3> d = x - mu;
            const Vector<T, 3> y = rotate_by_unit_quat_conjugate(q, d);

            // component log-pdf (normalization cancels in curvature ratio but needed for F, ∇F, ∇²F)
            const T md2 = lam.x * y.x * y.x + lam.y * y.y * y.y + lam.z * y.z * y.z;
            const T log_norm = -T(0.5) * T(3) * std::log(T(2.0) * T(M_PI))
                               - (std::log(sig.x) + std::log(sig.y) + std::log(sig.z));
            const T fi = wi * std::exp(log_norm - T(0.5) * md2); // f_i(x)

            // g_log_i = -A_i d, with A_i = R diag(λ) R^T
            const Matrix<T, 3, 3> Ai = build_A_from_quat_lambda(q, lam);
            const Vector<T, 3> gLog = Vector<T, 3>(
                -(Ai[0][0] * d.x + Ai[0][1] * d.y + Ai[0][2] * d.z),
                -(Ai[1][0] * d.x + Ai[1][1] * d.y + Ai[1][2] * d.z),
                -(Ai[2][0] * d.x + Ai[2][1] * d.y + Ai[2][2] * d.z)
            );

            // ∇F += f_i * g_log_i
            g += fi * gLog;

            // ∇²F += f_i * ( g_log_i g_log_i^T - A_i )
            Matrix<T, 3, 3> term = outer(gLog, gLog);
            term = sub(term, Ai);
            H += mul(term, fi);

            // F += f_i
            F += fi;
        }

        // Normal on the level set (same direction as ∇F)
        const T gnorm = norm(g);
        MixtureCurvature<T> out;
        out.F = F;
        out.grad = g;

        if (gnorm <= T(1e-20)) {
            // Degenerate: gradient too small → return zeros
            out.k1 = out.k2 = T(0);
            out.e1 = Vector<T, 3>(T(1), T(0), T(0));
            out.e2 = Vector<T, 3>(T(0), T(1), T(0));
            return out;
        }

        const Vector<T, 3> n = (T(1) / gnorm) * g;
        const Matrix<T, 3, 3> I = eye<T>();
        const Matrix<T, 3, 3> nnT = outer(n, n);
        const Matrix<T, 3, 3> P = sub(I, nnT);

        // H_T = P H P  (project Hessian to tangent plane)
        auto matmul = [](const Matrix<T, 3, 3> &A, const Matrix<T, 3, 3> &B) {
            Matrix<T, 3, 3> C;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    C[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
            return C;
        };
        const Matrix<T, 3, 3> PH = matmul(P, H);
        const Matrix<T, 3, 3> H_T = matmul(PH, P);

        // Build a tangent basis and assemble the 2x2 matrix of the shape operator:
        // S = -(1/||∇F||) * (t_i^T H_T t_j)
        Vector<T, 3> t1, t2;
        tangent_basis(n, t1, t2);
        auto quad = [&](const Vector<T, 3> &a, const Matrix<T, 3, 3> &B, const Vector<T, 3> &b)-> T {
            return a.x * (B[0][0] * b.x + B[0][1] * b.y + B[0][2] * b.z) +
                   a.y * (B[1][0] * b.x + B[1][1] * b.y + B[1][2] * b.z) +
                   a.z * (B[2][0] * b.x + B[2][1] * b.y + B[2][2] * b.z);
        };
        const T a = quad(t1, H_T, t1);
        const T b = quad(t1, H_T, t2);
        const T c = quad(t2, H_T, t2);

        // Eigenvalues of symmetric 2x2 [[a,b],[b,c]]
        const T tr = a + c;
        const T disc = (a - c) * (a - c) + T(4) * b * b;
        const T root = std::sqrt(disc >= T(0) ? disc : T(0));
        const T s1 = T(0.5) * (tr + root);
        const T s2 = T(0.5) * (tr - root);

        const T scale = T(-1) / gnorm; // minus sign from the definition of the shape operator
        out.k1 = s1 * scale;
        out.k2 = s2 * scale;

        // Principal directions in world: eigenvectors in (t1,t2) then lift
        Vector<T, 3> v1_2d, v2_2d;
        if (std::fabs(b) > T(1e-20)) {
            v1_2d = Vector<T, 3>(b, s1 - a, T(0));
            v2_2d = Vector<T, 3>(b, s2 - a, T(0));
        } else {
            v1_2d = Vector<T, 3>(T(1), T(0), T(0));
            v2_2d = Vector<T, 3>(T(0), T(1), T(0));
        }
        out.e1 = normalize(t1 * v1_2d.x + t2 * v1_2d.y);
        out.e2 = normalize(t1 * v2_2d.x + t2 * v2_2d.y);
        return out;
    }

    // ---------- apply Av without building A: Av = R * (Λ * (R^T v)) ----------
    template<typename T>
    CUDA_HOST_DEVICE
    inline Vector<T, 3> apply_Av(const Quaternion<T> &q, const Vector<T, 3> &lambda, const Vector<T, 3> &v_world) {
        const Vector<T, 3> u = rotate_by_unit_quat_conjugate(q, v_world); // u = R^T v
        const Vector<T, 3> w = v3(lambda.x * u.x, lambda.y * u.y, lambda.z * u.z); // w = Λ u
        return rotate_by_unit_quat(q, w); // Av = R w
    }

    // ---------- output ----------
    template<typename T>
    struct MixtureCurvaturesLean {
        T F; // mixture value
        Vector<T, 3> grad; // ∇F
        T k1, k2; // principal curvatures (k1 >= k2)
    };

    // ---------- core: F, ∇F, k1, k2 via only Av-products ----------
    template<typename T>
    CUDA_HOST_DEVICE
    MixtureCurvaturesLean<T> GaussianMixtureLevelSetCurvatures_Lean(
        const Vector<T, 3> &x,
        const Vector<T, 3> *means, // size N
        const Vector<T, 3> *log_sigmas, // size N (log σ)
        const Quaternion<T> *quats, // size N (unit; local→world)
        const T *weights, // size N (nullable => all ones)
        int N) {
        T F = T(0);
        Vector<T, 3> g = v3(T(0), T(0), T(0));

        // We also need H acting on t1, t2 later.
        // We'll compute H v via: H v = Σ_i f_i [ (g_i·v) g_i - A_i v ],
        // where g_i = -A_i d,  d = x - μ_i.  All uses Av-products.

        // First pass: accumulate F and gradF, also store per-i temporaries if N is small.
        // For a single-pass, we can recompute in the H·v stage to save memory.
        // Here we choose recompute (lean & cache-free).

        // Compute F, gradF
        for (int i = 0; i < N; ++i) {
            const T wi = weights ? weights[i] : T(1);
            const Vector<T, 3> mu = means[i];
            const Vector<T, 3> d = x - mu;

            const Vector<T, 3> sig = exp_vec(log_sigmas[i]); // σ
            const Vector<T, 3> invs = inv(sig);
            const Vector<T, 3> lam = v3(invs.x * invs.x, invs.y * invs.y, invs.z * invs.z); // 1/σ^2
            const Quaternion<T> q = quats[i];

            // y = R^T d
            const Vector<T, 3> y = rotate_by_unit_quat_conjugate(q, d);
            const T md2 = lam.x * y.x * y.x + lam.y * y.y * y.y + lam.z * y.z * y.z;

            const T log_norm = -T(0.5) * T(3) * std::log(T(2.0) * T(M_PI))
                               - (std::log(sig.x) + std::log(sig.y) + std::log(sig.z));
            const T fi = wi * std::exp(log_norm - T(0.5) * md2); // f_i(x)

            // g_i = ∇f_i = f_i * g_log_i, with g_log_i = -A_i d
            const Vector<T, 3> Ai_d = apply_Av(q, lam, d); // A_i d
            const Vector<T, 3> g_i = (-fi) * Ai_d; // f_i * (-A_i d)

            F += fi;
            g += g_i;
        }

        MixtureCurvaturesLean<T> out;
        out.F = F;
        out.grad = g;

        const T gnorm = norm(g);
        if (gnorm <= T(1e-20)) {
            out.k1 = out.k2 = T(0);
            return out;
        }

        const Vector<T, 3> n = (T(1) / gnorm) * g;
        Vector<T, 3> t1, t2;
        tangent_basis(n, t1, t2);

        // Helper to compute H·v on the fly (recomputes per-i terms; no matrices)
        auto H_times_v = [&](const Vector<T, 3> &v)-> Vector<T, 3> {
            Vector<T, 3> Hv = v3(T(0), T(0), T(0));
            for (int i = 0; i < N; ++i) {
                const T wi = weights ? weights[i] : T(1);
                const Vector<T, 3> mu = means[i];
                const Vector<T, 3> d = x - mu;

                const Vector<T, 3> sig = exp_vec(log_sigmas[i]);
                const Vector<T, 3> invs = inv(sig);
                const Vector<T, 3> lam = v3(invs.x * invs.x, invs.y * invs.y, invs.z * invs.z);
                const Quaternion<T> q = quats[i];

                // component value f_i(x)
                const Vector<T, 3> y = rotate_by_unit_quat_conjugate(q, d);
                const T md2 = lam.x * y.x * y.x + lam.y * y.y * y.y + lam.z * y.z * y.z;
                const T log_norm = -T(0.5) * T(3) * std::log(T(2.0) * T(M_PI))
                                   - (std::log(sig.x) + std::log(sig.y) + std::log(sig.z));
                const T fi = wi * std::exp(log_norm - T(0.5) * md2);

                // g_i = ∇f_i = -fi * (A_i d)
                const Vector<T, 3> Ai_d = apply_Av(q, lam, d);
                const Vector<T, 3> gi = (-fi) * Ai_d;

                // A_i v
                const Vector<T, 3> Aiv = apply_Av(q, lam, v);

                // accumulate: f_i [ (g_i·v) g_i - A_i v ]
                const T gi_dot_v = dot(gi, v);
                Hv += (gi * gi_dot_v) - (fi * Aiv);
            }
            return Hv;
        };

        // Build 2×2 symmetric matrix entries using H·t1, H·t2
        const Vector<T, 3> Ht1 = H_times_v(t1);
        const Vector<T, 3> Ht2 = H_times_v(t2);
        const T a = dot(t1, Ht1);
        const T b = dot(t1, Ht2); // = dot(t2, Ht1)
        const T c = dot(t2, Ht2);

        // Eigenvalues of [[a,b],[b,c]]
        const T tr = a + c;
        const T disc = (a - c) * (a - c) + T(4) * b * b;
        const T root = std::sqrt(disc >= T(0) ? disc : T(0));
        const T s1 = T(0.5) * (tr + root);
        const T s2 = T(0.5) * (tr - root);

        // Shape operator S = -(1/||∇F||) * P H P, restricted to tangent basis → scale 2×2 by (−1/||∇F||)
        const T scale = T(-1) / gnorm;
        out.k1 = s1 * scale;
        out.k2 = s2 * scale;
        return out;
    }
}
