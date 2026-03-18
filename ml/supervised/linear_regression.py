import numpy as np


class LinearRegressor:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.W = None
        self.B = 0.0

    def predict(self, X):
        return (np.matmul(X, self.W) + self.B).reshape(-1)

    def compute_cost(self, Y_hat, Y):
        # J = 1/(2m) * ||Y_hat - Y||^2
        Y, Y_hat = Y.reshape(-1), Y_hat.reshape(-1)
        m = Y.shape[0]
        err = Y_hat - Y
        return 1 / (2 * m) * np.matmul(err.T, err)

    def fit(self, X, Y):
        Y = Y.reshape(-1)
        m, n = X.shape

        self.W = np.zeros(n)
        self.B = 0.0

        for i in range(self.num_iterations):
            Y_hat = self.predict(X)
            err = Y_hat - Y

            dJ_dW = 1 / m * np.matmul(X.T, err)
            dJ_dB = 1 / m * np.sum(err)

            self.W -= self.learning_rate * dJ_dW
            self.B -= self.learning_rate * dJ_dB

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}, Cost: {self.compute_cost(Y_hat, Y):.6f}")

        return self

    def score(self, X, Y):
        # R² = 1 - SS_res / SS_tot
        Y = Y.reshape(-1)
        Y_hat = self.predict(X)
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        return 1 - ss_res / ss_tot
