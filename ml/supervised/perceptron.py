import numpy as np


class Perceptron:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def predict(self, X):
        return (np.matmul(X, self.w) + self.b >= 0).astype(int)

    def fit(self, X, y):
        m, n = X.shape
        y = y.reshape(-1, 1)

        self.w = np.zeros(n)
        self.b = 0.0

        for _ in range(self.num_iterations):
            err = self.predict(X).reshape(-1, 1) - y
            self.w -= self.learning_rate * np.sum(err * X, axis=0)
            self.b -= self.learning_rate * np.sum(err)

        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == y.reshape(-1))
