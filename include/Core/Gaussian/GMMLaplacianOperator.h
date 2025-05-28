//
// Created by alex on 28.05.25.
//

#ifndef ENGINE24_GMMLAPLACIANOPERATOR_H
#define ENGINE24_GMMLAPLACIANOPERATOR_H

#include "GaussianMixture.h"
#include "Eigen/Sparse"

namespace Bcg {
    class GMMLaplacianOperator {
    public:
        GMMLaplacianOperator(const GaussianMixture &gmm) : gmm(gmm) {}

        void compute_stiffness_matrix(float stiffness_factor = 1.0f);

        void compute_diagonal_stiffness_matrix();

        void compute_mass_matrix(float mass_factor = 1.0f);

        void compute_laplacian_matrix();

        void compute_lumped_mass_matrix();

        void compute_all(float stiffness_factor = 1.0f, float mass_factor = 1.0f);

        Eigen::SparseMatrix<float> S; // Stiffness matrix
        Eigen::SparseMatrix<float> D; // Diagonal stiffness matrix
        Eigen::SparseMatrix<float> M; // Mass matrix
        Eigen::SparseMatrix<float> L; // Laplacian matrix
        Eigen::SparseMatrix<float> M_lumped; // Lumped mass matrix
    private:
        const GaussianMixture &gmm;
        Eigen::Triplet<float> off_diagonal_stiffness_triplets;
        Eigen::Triplet<float> diagonal_stiffness_triplets;
        Eigen::Triplet<float> mass_triplets;
        Eigen::Triplet<float> lumped_mass_triplets;

    };
}

#endif //ENGINE24_GMMLAPLACIANOPERATOR_H
