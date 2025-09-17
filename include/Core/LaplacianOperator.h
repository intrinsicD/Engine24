// --- REQUIRED SPECTRA HEADERS ---
// Make sure the Spectra 'include' directory is in your project's include path

// For the standard problem (generalized = false)
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>

// For the generalized problem (generalized = true)
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
        Eigen::SparseMatrix<float> L;       // L = M^-1 * S
        Eigen::SparseMatrix<float> L_sym;   // L_sym = M^-1/2 * S * M^-1/2

        void build(const std::vector<Eigen::Triplet<float>> &S_triplets, const std::vector<Eigen::Triplet<float>> &M_triplets, long n) {
            S.resize(n, n);
            S.setFromTriplets(S_triplets.begin(), S_triplets.end());

            M.resize(n, n);
            M.setFromTriplets(M_triplets.begin(), M_triplets.end());

            // Clear derived matrices to ensure they are recomputed when needed
            M_inv.resize(0, 0);
            M_inv_sqrt.resize(0, 0);
            L.resize(0, 0);
            L_sym.resize(0, 0);
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

            std::vector<Eigen::Triplet<float>> triplets;
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

            std::vector<Eigen::Triplet<float>> triplets;
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
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<float>> solver(op, k, ncv, sigma);

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