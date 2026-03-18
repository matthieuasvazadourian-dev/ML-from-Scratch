import numpy as np


class SoftmaxRegression:
    def __init__(self, learning_rate, num_iterations, num_classes):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.theta = None  # shape (k, n)

    def predict_proba(self, X):
        z = np.matmul(X, self.theta.T)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerically stable softmax
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_cost(self, X, y):
        m = X.shape[0]
        p_hat = self.predict_proba(X)
        return -np.mean(np.log(p_hat[np.arange(m), y]))

    def fit(self, X, y):
        m, n = X.shape
        k = self.num_classes

        self.theta = np.zeros((k, n))

        # One-hot encode labels
        y_1hot = np.zeros((m, k))
        y_1hot[np.arange(m), y] = 1

        for i in range(self.num_iterations):
            p_hat = self.predict_proba(X)
            dJ_dtheta = 1 / m * np.matmul((p_hat - y_1hot).T, X)
            self.theta -= self.learning_rate * dJ_dtheta

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}, Cost: {self.compute_cost(X, y):.6f}")

        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == y.reshape(-1))
