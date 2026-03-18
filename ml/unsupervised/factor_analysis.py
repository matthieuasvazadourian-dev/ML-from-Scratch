import numpy as np


def _multivariate_gaussian(x, mu, sigma):
    n = x.shape[0]
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    exponent = -0.5 * (x - mu).T @ sigma_inv @ (x - mu)
    return 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(sigma_det)) * np.exp(exponent)


class FactorAnalysis:
    def __init__(self, num_iterations, dim_latent_variable, threshold):
        self.num_iterations = num_iterations
        self.dim_latent_variable = dim_latent_variable  # k
        self.threshold = threshold

        self.mu = None       # mean of X, shape (n,)
        self.lambda_ = None  # loading matrix, shape (n, k)
        self.psi = None      # diagonal noise covariance, shape (n, n)
        self.I = None        # identity, shape (k, k)

    def _E_step(self, X):
        mu_z_x = (self.lambda_.T @ np.linalg.inv(self.lambda_ @ self.lambda_.T + self.psi) @ (X - self.mu).T).T
        sigma_z_x = self.I - self.lambda_.T @ np.linalg.inv(self.lambda_ @ self.lambda_.T + self.psi) @ self.lambda_
        return mu_z_x, sigma_z_x

    def _M_step(self, X, mu_z_x, sigma_z_x):
        m, n = X.shape
        k = self.dim_latent_variable

        left = np.zeros((n, k))
        right = np.zeros((k, k))
        for i in range(m):
            A = (X[i] - self.mu).reshape(n, 1)
            B = mu_z_x[i].reshape(k, 1)
            left += A @ B.T
            right += B @ B.T + sigma_z_x
        self.lambda_ = left @ np.linalg.inv(right)

        total = np.zeros((n, n))
        for i in range(m):
            xc = (X[i] - self.mu).reshape(n, 1)
            mu_i = mu_z_x[i].reshape(k, 1)
            total += xc @ xc.T - xc @ mu_i.T @ self.lambda_.T - self.lambda_ @ mu_i @ xc.T \
                     + self.lambda_ @ (mu_i @ mu_i.T + sigma_z_x) @ self.lambda_.T
        self.psi = np.diag(np.diag(total / m))

    def fit(self, X):
        m, n = X.shape
        k = self.dim_latent_variable

        self.mu = np.mean(X, axis=0)
        self.I = np.eye(k)
        self.lambda_ = 0.01 * np.random.randn(n, k)
        self.psi = np.eye(n)

        for _ in range(self.num_iterations):
            mu_z_x, sigma_z_x = self._E_step(X)
            self._M_step(X, mu_z_x, sigma_z_x)

        return self

    def calculate_px(self, x):
        return _multivariate_gaussian(x, self.mu, self.lambda_ @ self.lambda_.T + self.psi)

    def check_anomaly(self, x_new):
        return self.calculate_px(x_new) < self.threshold
