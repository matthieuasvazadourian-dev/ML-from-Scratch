import numpy as np
from ml.unsupervised.kmeans import KMeans


def _multivariate_gaussian(x, mu, sigma):
    n = x.shape[0]
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    exponent = -0.5 * (x - mu).T @ sigma_inv @ (x - mu)
    return 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(sigma_det)) * np.exp(exponent)


class GMM:
    def __init__(self, num_gaussians, num_iterations, threshold):
        self.num_gaussians = num_gaussians
        self.num_iterations = num_iterations
        self.threshold = threshold

        self.means = None             # shape (k, n)
        self.covariances = None       # shape (k, n, n)
        self.multinomial_probs = None  # shape (k,) — mixture weights

    def _E_step(self, X):
        m, k = X.shape[0], self.num_gaussians
        weights = np.empty((m, k))
        for i in range(m):
            densities = np.array([
                _multivariate_gaussian(X[i], self.means[j], self.covariances[j]) * self.multinomial_probs[j]
                for j in range(k)
            ])
            weights[i] = densities / densities.sum()
        return weights

    def _M_step(self, X, weights):
        m, n = X.shape
        k = self.num_gaussians

        self.multinomial_probs = np.mean(weights, axis=0)
        denominator = np.sum(weights, axis=0)

        for j in range(k):
            w = weights[:, j]
            self.means[j] = (w @ X) / denominator[j]
            diff = X - self.means[j]
            self.covariances[j] = (w[:, None] * diff).T @ diff / denominator[j]

    def fit(self, X):
        m, n = X.shape
        k = self.num_gaussians

        self.multinomial_probs = np.full(k, 1 / k)

        # Initialise means with K-Means
        kmeans = KMeans(k).fit(X)
        self.means = kmeans.centroids.copy()

        self.covariances = np.empty((k, n, n))
        for j in range(k):
            assigned = X[kmeans.labels_ == j]
            self.covariances[j] = np.cov(X, rowvar=False) if len(assigned) < 2 else np.cov(assigned, rowvar=False)

        for _ in range(self.num_iterations):
            weights = self._E_step(X)
            self._M_step(X, weights)

        return self

    def calculate_px(self, x):
        return sum(
            _multivariate_gaussian(x, self.means[j], self.covariances[j]) * self.multinomial_probs[j]
            for j in range(self.num_gaussians)
        )

    def determine_anomaly(self, x_new):
        return self.calculate_px(x_new) < self.threshold
