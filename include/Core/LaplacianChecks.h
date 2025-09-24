// C++
// File: 'include/Core/LaplacianChecks.h'
#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <limits>

namespace Bcg {
    struct LaplacianDiagnostics {
        bool symmetric = false; //Should be true after build
        bool zero_row_sum = false; //Should be true after build
        bool nonpos_offdiag = false; //Should be true after build
        bool diagonally_dominant = false; //Should be true after build
        bool M_spd = false; // if M is provided
        int n = 0;
        int nnz = 0;
        int num_isolated = 0; // rows with all zeros
        double symmetry_norm = 0.0; // ||L - L^T||_F
        double ones_resid_inf = 0.0; // ||L * 1||_inf
        double min_diag = std::numeric_limits<double>::infinity();
        double max_offdiag = 0.0; // max offdiag value (should be <= 0)
    };

    template<typename Scalar>
    inline LaplacianDiagnostics AnalyzeLaplacian(const Eigen::SparseMatrix<Scalar> &L,
                                                 const Eigen::SparseMatrix<Scalar> *M = nullptr,
                                                 double tol = 1e-6) {
        using SpMat = Eigen::SparseMatrix<Scalar>;
        using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        LaplacianDiagnostics d;
        d.n = static_cast<int>(L.rows());
        d.nnz = static_cast<int>(L.nonZeros());

        // Use small relative tolerances
        double rtol = tol; // e.g. 1e-6

        // 1) Symmetry (relative)
        SpMat LT = SpMat(L.transpose());
        SpMat Skew = L - LT;
        double LnormF = (L).norm();
        d.symmetry_norm = Skew.norm();
        d.symmetric = (d.symmetry_norm <= std::max(1.0, LnormF) * rtol);

        // 2) Row sums (relative per row)
        d.zero_row_sum = true;
        d.num_isolated = 0;
        d.min_diag = std::numeric_limits<double>::infinity();
        d.max_offdiag = -std::numeric_limits<double>::infinity();

        bool all_offdiag_nonpos = true;
        bool diag_dom_ok = true;

        for (int i = 0; i < d.n; ++i) {
            double rowsum = 0.0;
            double rowL1 = 0.0;
            double diag = 0.0;
            bool any_nz = false;

            for (typename SpMat::InnerIterator it(L, i); it; ++it) {
                any_nz = true;
                int j = it.col();
                double v = static_cast<double>(it.value());
                rowsum += v;
                rowL1 += std::abs(v);
                if (i == j) {
                    diag += v;
                } else {
                    d.max_offdiag = std::max(d.max_offdiag, v);
                    if (v > rtol) all_offdiag_nonpos = false; // informational
                }
            }

            if (!any_nz) {
                d.num_isolated++;
                // Zero row sum trivially holds; continue
            }

            // Relative zero-row-sum check
            if (std::abs(rowsum) > std::max(1.0, rowL1) * rtol) d.zero_row_sum = false;

            d.min_diag = std::min(d.min_diag, diag);

            // Diagonal dominance (optional info; don't fail cotan on it)
            double offsum = rowL1 - std::abs(diag);
            if (diag + rtol < offsum) diag_dom_ok = false;
        }

        d.nonpos_offdiag = all_offdiag_nonpos;
        d.diagonally_dominant = diag_dom_ok;

        // 3) M SPD (if provided)
        if (M) {
            SpMat MT = SpMat(M->transpose());
            double MnormF = M->norm();
            bool M_sym = ((*M - MT).norm() <= std::max(1.0, MnormF) * rtol);

            bool pos_diag = true;
            for (int k = 0; k < M->rows(); ++k) {
                double mk = static_cast<double>(M->coeff(k, k));
                if (!(mk > 0.0)) {
                    pos_diag = false;
                    break;
                }
            }

            bool chol_ok = false;
            if (M_sym && pos_diag) {
                // Try sparse Cholesky (fast on diagonal/lumped M)
                Eigen::SimplicialLLT<SpMat> llt;
                llt.compute(*M);
                chol_ok = (llt.info() == Eigen::Success);
            }

            // Mark SPD if symmetry + positive diagonal + Cholesky succeeded
            d.M_spd = M_sym && pos_diag && chol_ok;
        }

        return d;
    }
} // namespace Bcg
