#pragma once
#include "Octree.h"
#include "GaussianUtils.h"
#include "AABBUtils.h"
#include "LaplacianOperator.h"
#include "LaplacianChecks.h"

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace Bcg {
    template<typename T>
    struct Gaussian {
        Eigen::Vector<T, 3> mu; // mean (world)
        Eigen::Vector<T, 3> log_sigma; // log stddevs (σ = exp(log_sigma))
        Eigen::Quaternion<T> q; // unit quaternion local->world
        T w = T(1); // optional mixture weight
    };

    struct Pair {
        int i, j;
    };

    template<typename T>
    inline Eigen::Vector<T, 3> decode_sigma_logstd(const Eigen::Vector<T, 3> &s_log) {
        return s_log.array().exp(); // σ = exp(s)
    }

    template<typename T>
    inline Eigen::Matrix<T, 3, 3> RotFromQuat(const Eigen::Quaternion<T> &q) {
        return q.normalized().toRotationMatrix();
    }

    template<typename T>
    inline Eigen::Matrix<T, 3, 3> SigmaFrom(const Gaussian<T> &g) {
        const Eigen::Matrix<T, 3, 3> R = RotFromQuat(g.q);
        const Eigen::Array<T, 3, 1> sig = g.log_sigma.array().exp(); // σ
        const Eigen::Array<T, 3, 1> sig2 = sig * sig; // σ^2
        return R * sig2.matrix().asDiagonal() * R.transpose();
    }

    template<typename T>
    inline Eigen::Matrix<T, 3, 3> PrecFrom(const Gaussian<T> &g) {
        const Eigen::Matrix<T, 3, 3> R = RotFromQuat(g.q);
        const Eigen::Array<T, 3, 1> sig = g.log_sigma.array().exp(); // σ
        const Eigen::Array<T, 3, 1> invsig2 = T(1) / (sig * sig);; // 1/σ^2
        return R * invsig2.matrix().asDiagonal() * R.transpose();
    }

    template<typename T>
    inline void TangentBasis(const Eigen::Vector<T, 3> &n,
                             Eigen::Vector<T, 3> &t1,
                             Eigen::Vector<T, 3> &t2) {
        if (std::abs(n.x()) > std::abs(n.z()))
            t1 = (Eigen::Vector<T, 3>(-n.y(), n.x(), T(0))).normalized();
        else
            t1 = (Eigen::Vector<T, 3>(T(0), -n.z(), n.y())).normalized();
        t2 = n.cross(t1);
    }

    template<typename T>
    inline T Det2x2(T a11, T a12, T a21, T a22) { return a11 * a22 - a12 * a21; }


    template<typename T>
    AABB<T> BuildAABB(const Eigen::Vector<T, 3> &mu,
                      const Eigen::Vector<T, 3> &scale,
                      const Eigen::Quaternion<T> &quat,
                      T k = 3.0) {
        // Normalize quaternion to be safe.
        Eigen::Quaternion<T> q = quat.normalized();
        const Eigen::Matrix<T, 3, 3> R = RotFromQuat(q);

        // k-scaled stddevs in local axes
        const Eigen::Vector<T, 3> ks = k * scale;

        // Per-axis squared extents in world: Δ_i^2 = sum_j (R_ij^2 * (kσ_j)^2)
        Eigen::Vector<T, 3> delta;
        const T ksx2 = ks.x() * ks.x();
        const T ksy2 = ks.y() * ks.y();
        const T ksz2 = ks.z() * ks.z();

        // Row 0 → x extent
        const T r00 = R(0, 0), r01 = R(0, 1), r02 = R(0, 2);
        delta.x() = std::sqrt(r00 * r00 * ksx2 + r01 * r01 * ksy2 + r02 * r02 * ksz2);

        // Row 1 → y extent
        const T r10 = R(1, 0), r11 = R(1, 1), r12 = R(1, 2);
        delta.y() = std::sqrt(r10 * r10 * ksx2 + r11 * r11 * ksy2 + r12 * r12 * ksz2);

        // Row 2 → z extent
        const T r20 = R(2, 0), r21 = R(2, 1), r22 = R(2, 2);
        delta.z() = std::sqrt(r20 * r20 * ksx2 + r21 * r21 * ksy2 + r22 * r22 * ksz2);

        AABB<T> box;
        Map(box.min) = mu - delta;
        Map(box.max) = mu + delta;

        // Optional: guard against zero-thickness due to tiny σ (helpful for robust tree building).
        const T eps = T(1e-9);
        for (int i = 0; i < 3; ++i) {
            if (box.max[i] - box.min[i] < eps) {
                box.min[i] -= T(0.5) * eps;
                box.max[i] += T(0.5) * eps;
            }
        }
        return box;
    }

    template<typename T>
    AABB<T> BuildAABB(const Gaussian<T> &g,
                      T k = 3.0) {
        return BuildAABB(g.mu, decode_sigma_logstd(g.log_sigma), g.q, k);
    }

    template<typename T>
    std::vector<AABB<T> > BuildAABBs(const std::vector<Gaussian<T> > &gaussians, T k = 3.0) {
        size_t size = gaussians.size();
        std::vector<AABB<T> > aabbs(size);
        for (size_t i = 0; i < size; i++) {
            aabbs[i] = BuildAABB(gaussians[i].mu, decode_sigma_logstd(gaussians[i].log_sigma), gaussians[i].q, k);
        }
        return aabbs;
    }

    template<typename T>
    std::vector<Pair> BuildOverlapPairs(const std::vector<Gaussian<T> > &G, Octree &octree, T k = 3.0) {
        std::vector<Pair> pairs;
        for (size_t i = 0; i < G.size(); i++) {
            std::vector<size_t> results;
            octree.query(BuildAABB(G[i], k), results);
            for (size_t j: results) {
                if (j <= i) continue; // avoid duplicates and self-pairs
                pairs.push_back({static_cast<int>(i), static_cast<int>(j)});
            }
        }
        return pairs;
    }


    template<typename T>
    struct PairTerms {
        T C_ij; // product coefficient
        Eigen::Vector<T, 3> mu_ij; // product mean
        Eigen::Matrix<T, 3, 3> Sigma_tilde_ij; // product covariance (3x3 SPD)
    };

    template<typename T>
    PairTerms<T> ComputePairProductParams(const Gaussian<T> &gi, const Gaussian<T> &gj) {
        const Eigen::Matrix<T, 3, 3> Si = SigmaFrom(gi);
        const Eigen::Matrix<T, 3, 3> Sj = SigmaFrom(gj);

        const Eigen::Matrix<T, 3, 3> Ai = PrecFrom(gi);
        const Eigen::Matrix<T, 3, 3> Aj = PrecFrom(gj);
        const Eigen::Matrix<T, 3, 3> Aij = Ai + Aj;

        Eigen::LLT<Eigen::Matrix<T, 3, 3> > llt(Aij);
        const Eigen::Matrix<T, 3, 3> SigT = llt.solve(Eigen::Matrix<T, 3, 3>::Identity());
        const Eigen::Vector<T, 3> muT = llt.solve(Ai * gi.mu + Aj * gj.mu);

        const Eigen::Matrix<T, 3, 3> Ssum = Si + Sj;
        Eigen::LLT<Eigen::Matrix<T, 3, 3> > lltS(Ssum);
        const auto L = lltS.matrixL();
        T logdetSsum = T(0);
        for (int d = 0; d < 3; ++d) logdetSsum += std::log(L(d, d));
        logdetSsum *= T(2);
        const Eigen::Vector<T, 3> d = gi.mu - gj.mu;
        const T quad = d.dot(lltS.solve(d));
        const T logC = -T(0.5) * (T(3) * std::log(T(2) * M_PI) + logdetSsum + quad);
        const T Cij = std::exp(logC);

        return {Cij, muT, SigT};
    }

    // ===================================================================
    // 2) ProjectorAt : freeze tangent on S_c at x̂ via short Newton from μ_ij
    // ===================================================================
    // A functor type that evaluates mixture F and grad at x
    template<typename T>
    using MixEval = std::function<void(const Eigen::Vector<T, 3> &x, T &F, Eigen::Vector<T, 3> &grad)>;

    template<typename T>
    struct FrozenTangent {
        Eigen::Vector<T, 3> xhat; // anchor on S_c
        Eigen::Vector<T, 3> n; // unit normal
        Eigen::Matrix<T, 3, 3> P; // projector I - nn^T
    };

    template<typename T>
    FrozenTangent<T> ProjectorAt(const PairTerms<T> &pij,
                                 T iso_c,
                                 const MixEval<T> &evalFgrad,
                                 int newton_iters = 2) {
        // start at μ_ij
        Eigen::Vector<T, 3> x = pij.mu_ij;

        for (int k = 0; k < newton_iters; ++k) {
            T F;
            Eigen::Vector<T, 3> g;
            evalFgrad(x, F, g);
            T g2 = g.squaredNorm();
            if (g2 <= T(1e-24)) break;
            // Newton step on level set F(x)=c
            x = x - ((F - iso_c) / g2) * g;
        }
        // final normal from mixture gradient
        T F;
        Eigen::Vector<T, 3> g;
        evalFgrad(x, F, g);
        Eigen::Vector<T, 3> n = g;
        const T gn = n.norm();
        if (gn > T(0)) n /= gn;
        else n = Eigen::Vector<T, 3>(1, 0, 0);

        const Eigen::Matrix<T, 3, 3> P = Eigen::Matrix<T, 3, 3>::Identity() - n * n.transpose();
        return FrozenTangent<T>{x, n, P};
    }

    // ===================================================================
    // 3) Mass entry  M_ij  (frozen tangent plane integral)
    // ===================================================================
    template<typename T>
    T Eval_M_ij_from_terms(const Gaussian<T> &gi, const Gaussian<T> &gj,
                           const PairTerms<T> &pij,
                           const FrozenTangent<T> &ft) {
        const auto &SigT = pij.Sigma_tilde_ij;
        const auto &xhat = ft.xhat;
        const auto &n = ft.n;

        // normal-direction 1D marginal at plane offset
        const T s2 = (n.transpose() * SigT * n)(0, 0); // variance along n
        const T dn = n.dot(pij.mu_ij - xhat); // signed offset (μ_ij to plane)
        const T N1D = T(1) / std::sqrt(T(2) * M_PI * s2) * std::exp(-T(0.5) * dn * dn / s2);

        // tangent 2D covariance Σ_TT = [ t_i^T Σ̃ t_j ]
        Eigen::Vector<T, 3> t1, t2;
        TangentBasis(n, t1, t2);
        const T a11 = t1.transpose() * SigT * t1;
        const T a12 = t1.transpose() * SigT * t2;
        const T a22 = t2.transpose() * SigT * t2;
        const T det2 = Det2x2(a11, a12, a12, a22);
        const T I2D = (T(2) * M_PI) * std::sqrt(det2); // ∫ over plane

        return pij.C_ij * N1D * I2D;
    }

    // ===================================================================
    // 4) Stiffness entry  A_ij  (plane expectation of gradient coupling)
    // ===================================================================
    template<typename T>
    T Eval_A_ij_from_terms(const Gaussian<T> &gi, const Gaussian<T> &gj,
                           const PairTerms<T> &pij,
                           const FrozenTangent<T> &ft) {
        // Precisions
        const Eigen::Matrix<T, 3, 3> Ai = PrecFrom(gi);
        const Eigen::Matrix<T, 3, 3> Aj = PrecFrom(gj);

        const auto &P = ft.P;
        const auto &xhat = ft.xhat;
        const auto &SigT = pij.Sigma_tilde_ij;

        // 1D normal factor (normal dir.) and 2D area factor (tangent)
        const T s2 = (ft.n.transpose() * SigT * ft.n)(0, 0);
        const T dn = ft.n.dot(pij.mu_ij - xhat);
        const T N1D = T(1) / std::sqrt(T(2) * M_PI * s2) * std::exp(-T(0.5) * dn * dn / s2);

        Eigen::Vector<T, 3> t1, t2;
        TangentBasis(ft.n, t1, t2);
        const T a11 = t1.transpose() * SigT * t1;
        const T a12 = t1.transpose() * SigT * t2;
        const T a22 = t2.transpose() * SigT * t2;
        const T det2 = Det2x2(a11, a12, a12, a22);
        const T I2D = (T(2) * M_PI) * std::sqrt(det2);

        // Embedded tangent covariance Σ_T = P Σ̃ P (rank-2 in R^3)
        const Eigen::Matrix<T, 3, 3> SigT_tan = P * SigT * P;

        // B = A_i P A_j  (symmetric on the tangent subspace)
        const Eigen::Matrix<T, 3, 3> B = Ai * P * Aj;

        // Expectation: E[(x-μ_i)^T B (x-μ_j)] over 2D Gaussian on plane with mean m = xhat
        const T trace_term = (B.cwiseProduct(SigT_tan)).sum(); // tr(B Σ_T)
        const Eigen::Vector<T, 3> di = xhat - gi.mu;
        const Eigen::Vector<T, 3> dj = xhat - gj.mu;
        const T mean_term = di.transpose() * B * dj;

        return pij.C_ij * N1D * I2D * (trace_term + mean_term);
    }

    template<typename T>
    struct LaplacianAssemblyOptions {
        T iso_c = T(1); // level-set value you integrate on (use your convention)
        bool symmetric_pair = true; // exploit A_ij = A_ji, M_ij = M_ji
        bool include_diagonal = true; // add i==j contributions (recommended)
        bool lump_mass = true; // diagonal lumping option
        bool normalize_rw = false; // build random-walk Laplacian L = I - M^{-1}A
        bool normalize_sym = false; // build symmetric L_sym = I - M^{-1/2} A M^{-1/2}
    };

    template<typename T>
    struct GaussianEvalEigen {
        T pdf;
        Eigen::Vector<T, 3> grad; /* add log_pdf/hess if needed */
    };

    template<typename T>
    GaussianEvalEigen<T> GaussianPdfGradHessEigen(const Eigen::Vector<T, 3> &x,
                                                  const Eigen::Vector<T, 3> &mu,
                                                  const Eigen::Vector<T, 3> &log_sigma,
                                                  const Eigen::Quaternion<T> &q) {
        // rotate world->local with q.conjugate()
        const auto qc = q.conjugate();
        const Eigen::Vector<T, 3> y = qc * (x - mu); // Eigen overload: q * v rotates

        const Eigen::Vector<T, 3> sigma = log_sigma.array().exp().matrix();
        const Eigen::Vector<T, 3> inv = sigma.cwiseInverse();
        const Eigen::Vector<T, 3> inv2 = inv.cwiseProduct(inv);

        const T md2 = (y.cwiseProduct(inv)).squaredNorm();
        const T log_norm = -T(0.5) * T(3) * std::log(T(2) * M_PI)
                           - (sigma.array().log().sum());
        const T logp = log_norm - T(0.5) * md2;
        const T p = std::exp(logp);

        // ∇_x log p = -A (x - mu); A = R diag(1/σ^2) R^T, but we do: g_log_world = q * ( -Λ y )
        const Eigen::Vector<T, 3> g_log_local = -y.cwiseProduct(inv2);
        const Eigen::Vector<T, 3> g_log_world = q * g_log_local;
        return {p, p * g_log_world};
    }

    template<typename T>
    LaplacianMatrices AssembleGaussianGalerkinLaplacian(
        const std::vector<Gaussian<T> > &G,
        const LaplacianAssemblyOptions<T> &opt,
        T c_value = T(1.0),
        T sigma_k_for_boxes = T(3.0)) {
        using Triplet = Eigen::Triplet<T>;
        const int n = static_cast<int>(G.size());

        std::vector<AABB<T> > gaussian_aabbs = BuildAABBs(G, sigma_k_for_boxes);
        Octree octree;
        octree.build(gaussian_aabbs, {Octree::SplitPoint::Median, true, 0.0f}, 32, 10);

        std::vector<Pair> pairs = BuildOverlapPairs(G, octree, sigma_k_for_boxes);

        std::vector<Triplet> trM;
        trM.reserve(pairs.size() * 2 + n);
        std::vector<Triplet> trA;
        trA.reserve(pairs.size() * 2 + n);

        // Accumulate diagonal if your formulas provide it directly.
        std::vector<T> diagM(n, T(0)), diagA(n, T(0));

        for (const auto &pr: pairs) {
            const int i = pr.i, j = pr.j;
            const auto &gi = G[i];
            const auto &gj = G[j];

            // Pairwise product parameters and frozen tangent projector
            auto pij = ComputePairProductParams(gi, gj);

            // 2) Provide a mixture evaluator F,grad (use your fast mixture code)
            MixEval<T> Fgrad = [&](const Eigen::Vector<T, 3> &x, T &F, Eigen::Vector<T, 3> &g) {
                // ... your mixture evaluation at x (sum_i w_i f_i(x), grad) ...
                F = T(0);
                g.setZero();

                // Query neighbors: tiny AABB around x (adjust eps if needed)
                const T eps = T(1e-6);
                AABB<T> qbox;
                qbox.min[0] = x.x() - eps;
                qbox.min[1] = x.y() - eps;
                qbox.min[2] = x.z() - eps;
                qbox.max[0] = x.x() + eps;
                qbox.max[1] = x.y() + eps;
                qbox.max[2] = x.z() + eps;

                std::vector<size_t> neigh;
                octree.query(qbox, neigh);

                for (size_t idx: neigh) {
                    const auto &gk = G[idx];
                    // Evaluate Gaussian pdf, grad in Eigen types (no covariance)
                    // Assumes you have this function taking Eigen types; see wrapper below if not.
                    auto e = GaussianPdfGradHessEigen<T>(x, gk.mu, gk.log_sigma, gk.q); // pdf, grad
                    F += gk.w * e.pdf;
                    g += gk.w * e.grad;
                }
            };


            // 3) Freeze tangent at level iso_c
            auto ft = ProjectorAt(pij, /*iso_c=*/c_value, Fgrad, /*newton_iters=*/2);

            // 4) Entries:
            T Mij = Eval_M_ij_from_terms(gi, gj, pij, ft);
            T Aij = Eval_A_ij_from_terms(gi, gj, pij, ft);

            if (i == j) {
                diagM[i] += Mij;
                diagA[i] += Aij;
                continue;
            }

            // Off-diagonals
            trM.emplace_back(i, j, Mij);
            trA.emplace_back(i, j, Aij);

            if (opt.symmetric_pair) {
                trM.emplace_back(j, i, Mij);
                trA.emplace_back(j, i, Aij);
            }

            // If your weak form naturally includes diagonal via i=j terms only, skip this.
            // Otherwise, if A_ii and M_ii are implied by symmetry/row-sum constraints, handle below.
        }
        Log::Info("Built system with {} pairs (of {} gaussians)", static_cast<int>(pairs.size()), n);

        // Diagonal triplets (from i==j terms)
        for (int i = 0; i < n; ++i) {
            if (diagM[i] != T(0)) trM.emplace_back(i, i, diagM[i]);
            if (diagA[i] != T(0)) trA.emplace_back(i, i, diagA[i]);
        }

        // Build sparse matrices
        LaplacianMatrices sys;
        sys.build(trA, trM, n);
        /*
        sys.M = Eigen::SparseMatrix<T>(n, n);
        sys.S = Eigen::SparseMatrix<T>(n, n);
        sys.M.setFromTriplets(trM.begin(), trM.end());
        sys.S.setFromTriplets(trA.begin(), trA.end());
        */

        // (Optional) **Mass lumping**: M_lumped = diag( sum_j M_ij )
        const T eps = T(1e-12);

        // 1) Sanitize A = sys.S (drop NaN/Inf/negatives), accumulate row sums
        std::vector<Eigen::Triplet<T> > trS;
        trS.reserve(sys.S.nonZeros() * 2 + n);
        Eigen::Matrix<T, -1, 1> rowSumA(n);
        rowSumA.setZero();

        size_t dropped_A = 0;
        for (int k = 0; k < sys.S.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(sys.S, k); it; ++it) {
                const int i = it.row();
                const int j = it.col();
                const T aij = it.value();

                if (!std::isfinite(static_cast<double>(aij)) || aij <= T(0)) {
                    ++dropped_A;
                    continue; // keep stiffness nonnegative and finite
                }
                if (i == j) {
                    // We rebuild the diagonal below from row sums; ignore here.
                    continue;
                }
                // Off-diagonal of Laplacian: S_ij = -A_ij
                trS.emplace_back(i, j, -aij);
                // Accumulate for diagonal: S_ii = sum_j A_ij
                rowSumA[i] += aij;
            }
        }

        // 2) Write Laplacian diagonal
        for (int i = 0; i < n; ++i) {
            const T di = std::isfinite(static_cast<double>(rowSumA[i])) ? std::max(rowSumA[i], eps) : eps;
            trS.emplace_back(i, i, di);
        }

        // Replace S by the proper Laplacian
        Eigen::SparseMatrix<T> S_lap(n, n);
        S_lap.setFromTriplets(trS.begin(), trS.end(),
                              [](const T &a, const T &b) { return a + b; });
        S_lap.makeCompressed();
        sys.S.swap(S_lap);

        // 3) Mass matrix: lump to strictly positive diagonal (sanitized)
        Eigen::Matrix<T, -1, 1> rowSumM(n);
        rowSumM.setZero();

        size_t dropped_M = 0;
        for (int k = 0; k < sys.M.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(sys.M, k); it; ++it) {
                const int i = it.row();
                const int j = it.col();
                const T mij = it.value();
                if (!std::isfinite(static_cast<double>(mij)) || mij < T(0)) {
                    ++dropped_M;
                    continue;
                }
                rowSumM[i] += mij;
                if (!opt.lump_mass && i == j) {
                    // If not lumping, we still want to keep any good diagonal entries
                    // (we'll rebuild below with sanitization).
                }
            }
        }

        std::vector<Eigen::Triplet<T> > trM_clean;
        trM_clean.reserve(n);
        if (opt.lump_mass) {
            for (int i = 0; i < n; ++i) {
                const T di = std::isfinite(static_cast<double>(rowSumM[i])) ? std::max(rowSumM[i], eps) : eps;
                trM_clean.emplace_back(i, i, di);
            }
        } else {
            // Non-lumped path: keep only positive, finite diagonals; zero others to eps.
            // (Safer alternative is to always lump; generalized eigensolvers prefer SPD M.)
            for (int i = 0; i < n; ++i) {
                const T di = std::isfinite(static_cast<double>(rowSumM[i])) && rowSumM[i] > T(0) ? rowSumM[i] : eps;
                trM_clean.emplace_back(i, i, di);
            }
        }

        Eigen::SparseMatrix<T> M_pos(n, n);
        M_pos.setFromTriplets(trM_clean.begin(), trM_clean.end());
        M_pos.makeCompressed();
        sys.M.swap(M_pos);

        // 4) Log diagnostics
        Log::Info(fmt::format(
            "Sanitized A: dropped {} entries; built Laplacian with nnz = {}",
            dropped_A, sys.S.nonZeros()));
        Log::Info(fmt::format(
            "Sanitized M: dropped {} entries; {}lumped, nnz = {}",
            dropped_M, opt.lump_mass ? "" : "non-", sys.M.nonZeros()));

        // 5) Final summary
        Log::Info(fmt::format(
            "Computed Laplacian matrices: S size = {}x{}, nnz = {}; M size = {}x{}, nnz = {}",
            sys.S.rows(), sys.S.cols(), sys.S.nonZeros(),
            sys.M.rows(), sys.M.cols(), sys.M.nonZeros()
        ));

        return sys;
    }
}
