import numpy as np


# Collaborative filtering via matrix factorisation with mean normalisation.
# Learns user weights W, user biases B, and item features X such that
# the predicted rating for user u on item i is: W[u] @ X[i] + B[u] + mean_R[u]

class CollaborativeFiltering:
    def __init__(self, learning_rate, num_iterations, lambda_, num_features_items):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.num_features_items = num_features_items

        self.W = None       # User weight matrix, shape (n_users, k)
        self.B = None       # User biases, shape (n_users,)
        self.X = None       # Item feature matrix, shape (n_items, k)
        self.mean_R = None  # Per-user mean rating, shape (n_users,)

    def predict(self):
        # Full rating matrix prediction with mean normalisation
        return (self.X @ self.W.T + self.B + self.mean_R).T

    def fit(self, R):
        # R is the rating matrix, shape (n_users, n_items); 0 means unrated
        users, items = np.where(R != 0)
        M = (R != 0).astype(int)

        self.mean_R = np.sum(R, axis=1) / np.maximum(np.sum(M, axis=1), 1)
        R_centered = R - self.mean_R[:, None]
        ratings = R_centered[users, items]

        n_users, n_items = R_centered.shape
        k = self.num_features_items
        N = len(ratings)

        rng = np.random.default_rng(0)
        self.W = 0.1 * rng.standard_normal((n_users, k))
        self.X = 0.1 * rng.standard_normal((n_items, k))
        self.B = np.zeros(n_users)

        for i in range(self.num_iterations):
            for j in rng.permutation(N):
                u, it, r = users[j], items[j], ratings[j]
                err = self.W[u] @ self.X[it] + self.B[u] - r

                W_u_old = self.W[u].copy()
                self.W[u] -= self.learning_rate * (err * self.X[it] + self.lambda_ * self.W[u])
                self.B[u] -= self.learning_rate * err
                self.X[it] -= self.learning_rate * err * W_u_old

            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{self.num_iterations}")

        return self
