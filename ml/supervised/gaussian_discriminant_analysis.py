import numpy as np


# Gaussian Discriminant Analysis (GDA) with a shared covariance matrix.
# A generative learning algorithm: learns P(X|Y) and P(Y), then uses Bayes' rule for P(Y|X).

class GDA:
    def __init__(self):
        self.phi = None     # P(Y=1), scalar
        self.mu0 = None     # Mean of class 0, shape (n,)
        self.mu1 = None     # Mean of class 1, shape (n,)
        self.sigma_ = None  # Shared covariance matrix, shape (n, n)

    def _multivariate_gaussian(self, x, mu):
        n = x.shape[0]
        sigma_inv = np.linalg.inv(self.sigma_)
        sigma_det = np.linalg.det(self.sigma_)
        exponent = -0.5 * (x - mu).T @ sigma_inv @ (x - mu)
        return 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(sigma_det)) * np.exp(exponent)

    def fit(self, X, Y):
        m, n = X.shape

        self.phi = np.mean(Y == 1)
        self.mu0 = np.mean(X[Y == 0], axis=0)
        self.mu1 = np.mean(X[Y == 1], axis=0)

        self.sigma_ = np.zeros((n, n))
        for i in range(m):
            mu = self.mu1 if Y[i] == 1 else self.mu0
            diff = (X[i] - mu).reshape(n, 1)
            self.sigma_ += diff @ diff.T
        self.sigma_ /= m

        return self

    def predict(self, X):
        m = X.shape[0]
        p_Y0, p_Y1 = 1 - self.phi, self.phi

        p_X_Y0 = np.array([self._multivariate_gaussian(X[i], self.mu0) for i in range(m)])
        p_X_Y1 = np.array([self._multivariate_gaussian(X[i], self.mu1) for i in range(m)])

        return np.argmax(np.stack([p_Y0 * p_X_Y0, p_Y1 * p_X_Y1]), axis=0)

    def score(self, X, Y):
        return np.mean(self.predict(X) == Y.reshape(-1))
