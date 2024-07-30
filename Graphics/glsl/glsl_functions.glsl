mat3 computeCovarianceMatrix(vec3 queryPoint, int k, int nearestPoints[], vec3 points[]) {
    mat3 covariance = mat3(0.0);
    for (int i = 0; i < k; i++) {
        vec3 knnPoint = points[nearestPoints[i]];

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
    return covariance;
}

vec3 computeEigenvalues(mat3 matrix) {
    // Compute the eigenvalues and eigenvectors of the covariance matrix
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
        // The covariance matrix is already diagonal.
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

    // -1 <= r <= 1
    float phi;
    if (r <= -1.0)
    phi = 3.14159265358979323846 / 3.0;
    else if (r >= 1.0)
    phi = 0.0;
    else
    phi = acos(r) / 3.0;

    // The eigenvalues of the covariance matrix
    float eig1 = q + 2.0 * p * cos(phi);
    float eig3 = q + 2.0 * p * cos(phi + (2.0 * 3.14159265358979323846 / 3.0));
    float eig2 = 3.0 * q - eig1 - eig3;  // since trace(A) = eig1 + eig2 + eig3
    return vec3(eig1, eig2, eig3);
}

vec3 computeEigenvector(mat3 matrix, float eigenvalue) {
    mat3 A = matrix - eigenvalue * mat3(1.0);
    return normalize(cross(A[0], A[1]));
}

mat3 computeEigenvector(mat3 matrix, vec3 eigenvalues) {
    // Find the eigenvector corresponding to the smallest eigenvalue (eig3)
    mat3 eigenvectors;
    eigenvectors[0] = computeEigenvector(matrix, eigenvalues[0]);
    eigenvectors[1] = computeEigenvector(matrix, eigenvalues[1]);
    eigenvectors[2] = computeEigenvector(matrix, eigenvalues[2]);
    return eigenvectors;
}