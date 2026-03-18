import numpy as np
from ml.preprocessing import StandardScaler


class PCA:
    def __init__(self, num_principal_components):
        self.num_principal_components = num_principal_components
        self.principal_components = None   # shape (n, N) — top N eigenvectors
        self.associated_eigenvalues = None  # shape (N,)
        self.scaler = None

    def fit(self, X):
        N = self.num_principal_components

        # Standardise before computing covariance
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        cov_matrix = np.cov(X_scaled, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # eigh returns eigenvalues in ascending order — take the last N
        self.principal_components = eigenvectors[:, -N:]
        self.associated_eigenvalues = eigenvalues[-N:]

        return self

    def transform(self, X):
        return self.scaler.transform(X) @ self.principal_components

    def inverse_transform(self, X_reduced):
        # Project back to original (scaled) space, then undo standardisation
        X_scaled_approx = X_reduced @ self.principal_components.T
        return self.scaler.inverse_transform(X_scaled_approx)
