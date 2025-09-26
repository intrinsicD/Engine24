#pragma once

#include "LaplacianChecks.h"

#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>

#include <Spectra/MatOp/SymShiftInvert.h>     // The key operator for (A - sB)^-1
#include <Spectra/MatOp/SparseSymMatProd.h>    // The key operator for B*v
#include <Spectra/SymGEigsShiftSolver.h>      // The correct solver class

namespace Bcg {
    // ... (The LaplacianMatrices struct and its methods remain unchanged and correct) ...
    struct LaplacianMatrices {
        // --- Fundamental Matrices (must be provided by user) ---
        Eigen::SparseMatrix<float> S;
        Eigen::SparseMatrix<float> M;

        // --- Derived Matrices (lazily computed) ---
        Eigen::SparseMatrix<float> M_inv;
        Eigen::SparseMatrix<float> M_inv_sqrt;
        Eigen::SparseMatrix<float> L; // L = M^-1 * S
        Eigen::SparseMatrix<float> L_sym; // L_sym = M^-1/2 * S * M^-1/2

        void build(const std::vector<Eigen::Triplet<float> > &S_triplets,
                   const std::vector<Eigen::Triplet<float> > &M_triplets, long n) {
            S.resize(n, n);
            S.setFromTriplets(S_triplets.begin(), S_triplets.end(),
                              [](const float &a, const float &b) { return a + b; });

            M.resize(n, n);
            M.setFromTriplets(M_triplets.begin(), M_triplets.end(),
                              [](const float &a, const float &b) { return a + b; });

            // Clear derived matrices to ensure they are recomputed when needed
            M_inv.resize(0, 0);
            M_inv_sqrt.resize(0, 0);
            L.resize(0, 0);
            L_sym.resize(0, 0);

            S.makeCompressed();
            M.makeCompressed();

            auto result = AnalyzeLaplacian(S, &M);
            if (!result.symmetric) {
                Log::Error(fmt::format("Laplacian matrix S is not symmetric. Symmetry norm: {}", result.symmetry_norm));
            }
            if (!result.zero_row_sum) {
                Log::Error(fmt::format("Laplacian matrix S does not have zero row sums. Row sum infinity norm: {}",
                                       result.ones_resid_inf));
            }
            if (!result.nonpos_offdiag) {
                Log::Error(fmt::format(
                    "Laplacian matrix S has positive off-diagonal entries. Max off-diagonal value: {}",
                    result.max_offdiag));
            }
            if (!result.diagonally_dominant) {
                Log::Error(fmt::format(
                    "Laplacian matrix S is not diagonally dominant. Min diagonal: {}, Max off-diagonal sum: {}",
                    result.min_diag, result.max_offdiag));
            }
            if (!result.M_spd) {
                Log::Error(fmt::format("Laplacian matrix M is not sparse."));
            }
        }

        // --- 'Exists' Checkers ---
        bool L_exists() const { return L.rows() > 0; }

        bool L_sym_exists() const { return L_sym.rows() > 0; }

        bool M_inv_exists() const { return M_inv.rows() > 0; }

        bool M_inv_sqrt_exists() const { return M_inv_sqrt.rows() > 0; }

        // --- 'Create' Methods for Lazy Evaluation ---
        void create_M_inv(bool force = false) {
            if (!force && M_inv_exists()) return;
            if (M.rows() == 0) throw std::runtime_error("Mass matrix M is not initialized.");

            std::vector<Eigen::Triplet<float> > triplets;
            triplets.reserve(M.nonZeros());
            for (int k = 0; k < M.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(M, k); it; ++it) {
                    if (it.row() == it.col() && it.value() != 0.0f) {
                        triplets.emplace_back(it.row(), it.col(), 1.0f / it.value());
                    }
                }
            }
            M_inv.resize(M.rows(), M.cols());
            M_inv.setFromTriplets(triplets.begin(), triplets.end());
        }

        void create_M_inv_sqrt(bool force = false) {
            if (!force && M_inv_sqrt_exists()) return;
            if (M.rows() == 0) throw std::runtime_error("Mass matrix M is not initialized.");

            std::vector<Eigen::Triplet<float> > triplets;
            triplets.reserve(M.nonZeros());
            for (int k = 0; k < M.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(M, k); it; ++it) {
                    if (it.row() == it.col() && it.value() > 0.0f) {
                        triplets.emplace_back(it.row(), it.col(), 1.0f / std::sqrt(it.value()));
                    }
                }
            }
            M_inv_sqrt.resize(M.rows(), M.cols());
            M_inv_sqrt.setFromTriplets(triplets.begin(), triplets.end());
        }

        void create_L(bool force = false) {
            if (!force && L_exists()) return;
            if (S.rows() == 0) throw std::runtime_error("Stiffness matrix S is not initialized.");
            create_M_inv(force);
            L = M_inv * S;
        }

        void create_L_sym(bool force = false) {
            if (!force && L_sym_exists()) return;
            if (S.rows() == 0) throw std::runtime_error("Stiffness matrix S is not initialized.");
            create_M_inv_sqrt(force);
            L_sym = M_inv_sqrt * S * M_inv_sqrt;
        }
    };

    // ... (GetTransitionMatrix and EigenDecompositionResult remain unchanged) ...
    struct EigenDecompositionResult {
        Eigen::Matrix<float, -1, -1> evecs;
        Eigen::Vector<float, -1> evals;
    };


    /**
     * @brief Computes the first k eigenvalues/vectors of the Laplacian using a sparse solver.
     * THIS IS THE FINAL, CORRECT IMPLEMENTATION based on Spectra's documented design patterns.
     */
    inline EigenDecompositionResult EigenDecompositionSparse(
        LaplacianMatrices &matrices,
        int k,
        bool generalized = true,
        float sigma = 0.0f) {

        if (k <= 0) throw std::invalid_argument("Number of eigenvalues k must be positive.");
        if (matrices.S.rows() == 0) throw std::runtime_error("Stiffness matrix S not initialized.");
        if (k >= matrices.S.rows()) throw std::invalid_argument("k must be smaller than the matrix size.");

        // Basic sanity: avoid exact-singular K = S - σ M at σ = 0 for Laplacians
        if (generalized && std::abs(sigma) < 1e-12f) {
            // pick a small positive shift away from the nullspace
            sigma = 1e-3f;
            Log::Info("Adjusted sigma to {} to avoid singular (S - sigma*M).", sigma);
        }

        // Quick finite/symmetry checks to fail fast before factorization
        auto chk = AnalyzeLaplacian(matrices.S, &matrices.M);
        if (!chk.symmetric || std::isnan(chk.symmetry_norm) || std::isinf(chk.symmetry_norm)) {
            throw std::invalid_argument("Matrix S is not finite/symmetric; cannot factorize (S - sigma*M).");
        }
        // M should be SPD (diagonal lumping recommended)
        if (!chk.M_spd) {
            throw std::invalid_argument("Matrix M is not SPD; enable mass lumping or fix assembly.");
        }

        EigenDecompositionResult result;
        int ncv = std::min((int) matrices.S.rows() - 1, 2 * k + 1);
        if (ncv <= k) throw std::invalid_argument("k is too large for the matrix size, cannot set ncv > k.");

        if (generalized) {
            // --- GENERALIZED PATH: Solves S*v = lambda*M*v ---
            // As per the Spectra documentation for SymGEigsShiftSolver, we need two operators.

            // 1. An operator for the B matrix (our M) to compute M*v.
            //    We use the standard SparseSymMatProd wrapper.
            Spectra::SparseSymMatProd<float> opM(matrices.M);

            // 2. An operator for the shift-solve operation (S - sigma*M)^-1 * v.
            //    The correct wrapper for this is SymShiftInvert.
            //    Since S and M are both sparse, the template is <float, Sparse, Sparse>.
            Spectra::SymShiftInvert<float, Eigen::Sparse, Eigen::Sparse> opInv(matrices.S, matrices.M);

            // 3. The solver that uses the shift-invert mode for generalized symmetric problems.
            Spectra::SymGEigsShiftSolver<
                        Spectra::SymShiftInvert<float, Eigen::Sparse, Eigen::Sparse>,
                        Spectra::SparseSymMatProd<float>,
                        Spectra::GEigsMode::ShiftInvert>
                    solver(opInv, opM, k, ncv, sigma);

            solver.init();
            int nconv = solver.compute(Spectra::SortRule::LargestMagn); // Shift-invert finds eigenvalues of (A-sB)^-1B
            // so largest magnitude eigenvalues of this op
            // correspond to eigenvalues of original problem closest to sigma.

            if (solver.info() != Spectra::CompInfo::Successful) {
                throw std::runtime_error("Sparse generalized eigendecomposition failed to converge.");
            }
            result.evals = solver.eigenvalues();
            result.evecs = solver.eigenvectors();
        } else {
            // --- STANDARD PATH: Solves L_sym*u = lambda*u ---
            // This path was already correct. It needs one operator for (L_sym - sigma*I)^-1 * v.
            matrices.create_L_sym();

            Spectra::SparseSymShiftSolve<float> op(matrices.L_sym);
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<float> > solver(op, k, ncv, sigma);

            solver.init();
            int nconv = solver.compute(Spectra::SortRule::LargestMagn);

            if (solver.info() != Spectra::CompInfo::Successful) {
                throw std::runtime_error("Sparse standard eigendecomposition failed to converge.");
            }
            result.evals = solver.eigenvalues();
            result.evecs = solver.eigenvectors();
        }
        return result;
    }
}
