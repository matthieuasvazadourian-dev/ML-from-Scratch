import numpy as np
from ml.unsupervised.pca import PCA


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class ICA:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None             # Unmixing matrix, shape (n, n)
        self.whitened_data = None  # shape (m, n)

    def fit(self, X):
        m, n = X.shape

        # Whiten the data via PCA (removes correlations and normalises variance)
        pca = PCA(n).fit(X)
        Z = pca.transform(X)
        self.whitened_data = Z @ np.sqrt(np.linalg.inv(np.diag(pca.associated_eigenvalues)))

        self.W = np.eye(n)

        # Stochastic gradient ascent on the ICA log-likelihood
        indices = np.arange(m)
        rng = np.random.default_rng(0)

        for _ in range(self.epochs):
            rng.shuffle(indices)
            for i in indices:
                x_i = self.whitened_data[i].reshape(n, 1)
                # Gradient: d/dW log|det W| - 2*sigmoid(Wx)*x^T
                dL_dW = np.eye(n) - 2 * _sigmoid(self.W @ x_i) @ x_i.T + np.linalg.inv(self.W.T)
                self.W += self.learning_rate * dL_dW

        return self

    def recover_signals(self):
        return (self.W @ self.whitened_data.T).T
