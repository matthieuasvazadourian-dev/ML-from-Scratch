import numpy as np


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip for numerical stability


class LogisticRegressor:
    def __init__(self, learning_rate, num_iterations, threshold=0.5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.W = None
        self.B = None

    def predict_proba(self, X):
        return _sigmoid(np.matmul(X, self.W) + self.B)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def compute_cost(self, Y_hat, Y):
        return np.mean(-Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat))

    def fit(self, X, Y):
        Y = Y.reshape(-1)
        m, n = X.shape

        self.W = np.zeros(n)
        self.B = 0.0

        for i in range(self.num_iterations):
            Y_hat = self.predict_proba(X)
            err = Y_hat - Y

            dJ_dW = 1 / m * np.matmul(X.T, err)
            dJ_dB = 1 / m * np.sum(err)

            self.W -= self.learning_rate * dJ_dW
            self.B -= self.learning_rate * dJ_dB

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}, Cost: {self.compute_cost(Y_hat, Y):.6f}")

        return self

    def score(self, X, Y):
        return np.mean(self.predict(X) == Y.reshape(-1))
